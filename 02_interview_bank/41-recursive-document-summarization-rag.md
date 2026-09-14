# 41 — Recursive Document Summarization RAG

> A multi-level summarization hierarchy built offline from the original corpus — documents → section summaries → document summaries → corpus summaries — where query routing at inference time selects the right abstraction level rather than always retrieving raw chunks.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
Corpus
  │
  ▼
Chunk-level Summaries (Level 0: raw chunks)
  │  summarize groups of chunks
  ▼
Section-level Summaries (Level 1)
  │  summarize groups of sections
  ▼
Document-level Summaries (Level 2)
  │  summarize groups of documents
  ▼
Corpus-level Summary (Level 3)

  (4-level tree built along the document's natural
   structure — NOT clustering, unlike RAPTOR)

────────────────────── query time ──────────────────────

Query
  │
  ▼
Level Router
  (picks which tier to search based on query scope:
   overview → L2–3, section → L1, chunk → L0, multi → all)
  │
  ▼
Retriever (fetches from the chosen level)
  │
  ▼
Generator
```

### Key Components

| Component | Responsibility |
|---|---|
| Recursive Summarizer (LLM) | Generates faithful summaries bottom-up: chunks → sections → documents → corpus |
| 4-level Summary Tree Store | Persists all levels of nodes (chunk/section/document/corpus) with parent/child links and embeddings |
| Level Router | Classifies each query's required abstraction level and selects which tier(s) to search |
| Retriever | Runs similarity search against the nodes at the routed level(s) |
| Generator | Produces the final answer from the retrieved nodes (optionally after drill-down) |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Summarization LLM | Any LLM; GPT-4o-mini or Claude Haiku for cost-efficient recursive summarization |
| Vector store | Vector DB with level metadata (similar infra to RAPTOR, but hierarchy follows document structure rather than semantic clustering) |

---

## Q1. What is Recursive Document Summarization RAG and what problem does it solve? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Recursive Document Summarization RAG builds a multi-level tree of summaries over the corpus at index time, following the document's own structure top-down rather than clustering chunks by semantic similarity:

```
Level 3: Corpus summary (1 per collection — "what does this knowledge base contain?")
Level 2: Document summary (1 per document — "what is this document about?")
Level 1: Section summaries (1 per section — "what does this section cover?")
Level 0: Original chunks (raw paragraphs/sentences — the actual source text)
```

The problem it solves: a flat retrieval system only ever operates at one level of granularity (raw chunks), which is well-suited to specific factual queries but poorly suited to broad, orientation-style questions ("what does this report cover overall?") — those questions either retrieve a semi-random handful of chunks that don't collectively answer the question, or require stuffing far more raw text into context than necessary. By pre-building summaries at every level of the document's natural structure, a query can be answered from whichever granularity actually matches its scope, rather than forcing every query through the same chunk-level retrieval path.

</details>

---

## Q2. What is the single distinctive mechanism that separates this architecture from RAPTOR? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The distinctive mechanism is **top-down summarization within document boundaries**, in contrast to RAPTOR's (#13) **bottom-up clustering across document boundaries**. Every node in this architecture's tree — a section summary, a document summary, the corpus summary — belongs to exactly one source document (or the whole corpus at the top), preserving provenance at every level. RAPTOR's cluster nodes, by contrast, group semantically similar chunks regardless of which document they came from, so a single RAPTOR summary node can blend content from several unrelated source documents.

This single structural difference is what determines which architecture fits which navigation pattern: "what does document A say about X?" is a natural fit for this architecture's document-scoped hierarchy, while "what do all documents say about topic Y?" is what RAPTOR's cross-document clustering is built for (Q6, Q7).

</details>

---

## Q3. Walk through the end-to-end architecture, from raw corpus to a query-time answer. `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

```
Corpus
  │
  ▼
Chunk-level Summaries (Level 0: raw chunks)
  │  summarize groups of chunks
  ▼
Section-level Summaries (Level 1)
  │  summarize groups of sections
  ▼
Document-level Summaries (Level 2)
  │  summarize groups of documents
  ▼
Corpus-level Summary (Level 3)

────────────── query time ──────────────
Query → Level Router (overview→L2-3, section→L1, chunk→L0, multi→all)
      → Retriever (fetches from the chosen level)
      → Generator
```

Building the tree happens once, offline (Q4): each level's summaries are generated from the level below it, so section summaries come from their chunks, document summaries come from their sections' summaries, and the single corpus summary comes from every document's summary. At query time, a lightweight router (Q5) classifies the query's required abstraction level *before* any retrieval happens, so retrieval only ever searches the level(s) that actually match the query's scope — a broad query never wastes its search on raw chunks, and a specific factual query never wastes it on an overly-compressed summary.

</details>

---

## Q4. How do you build the multi-level summary tree in code? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Each `SummaryNode` records its level, its content, and explicit parent/child links so the tree can be navigated in either direction (routing down from a level, or drilling down from a summary to its source chunks):

```python
@dataclass
class SummaryNode:
    node_id: str
    level: int                    # 0 = chunk, 1 = section, 2 = document, 3 = corpus
    content: str
    parent_id: Optional[str]
    children_ids: list[str]
    doc_id: str
    section_title: Optional[str] = None
    embedding: Optional[list[float]] = None

def summarize(texts: list[str], context: str = "") -> str:
    joined = "\n\n---\n\n".join(texts)
    resp = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=512,
        system=("Write a dense, faithful summary of the provided text. "
                 "Preserve key facts, figures, and named entities. "
                 "Do not add information not present in the text."
                 + (f" Context: {context}" if context else "")),
        messages=[{"role": "user", "content": joined}])
    return resp.content[0].text

def build_summary_tree(documents: list[dict], embed_fn) -> list[SummaryNode]:
    all_nodes, corpus_doc_summaries = [], []
    for doc in documents:
        doc_section_summaries = []
        for section in doc["sections"]:
            chunk_nodes = [SummaryNode(node_id=f"{doc['id']}::{section['title']}::chunk_{i}",
                                        level=0, content=c, parent_id=f"{doc['id']}::{section['title']}::summary",
                                        children_ids=[], doc_id=doc["id"], section_title=section["title"],
                                        embedding=embed_fn(c))
                           for i, c in enumerate(section["chunks"])]
            all_nodes.extend(chunk_nodes)

            section_summary_text = summarize(section["chunks"], context=f"Section '{section['title']}' from document '{doc['title']}'")
            section_node = SummaryNode(node_id=f"{doc['id']}::{section['title']}::summary", level=1,
                                        content=section_summary_text, parent_id=f"{doc['id']}::summary",
                                        children_ids=[n.node_id for n in chunk_nodes], doc_id=doc["id"],
                                        section_title=section["title"], embedding=embed_fn(section_summary_text))
            all_nodes.append(section_node)
            doc_section_summaries.append(section_summary_text)

        doc_summary_text = summarize(doc_section_summaries, context=f"Document: '{doc['title']}'")
        doc_node = SummaryNode(node_id=f"{doc['id']}::summary", level=2, content=doc_summary_text,
                                parent_id="corpus::summary",
                                children_ids=[f"{doc['id']}::{s['title']}::summary" for s in doc["sections"]],
                                doc_id=doc["id"], embedding=embed_fn(doc_summary_text))
        all_nodes.append(doc_node)
        corpus_doc_summaries.append(doc_summary_text)

    corpus_summary_text = summarize(corpus_doc_summaries, context="Full document corpus")
    all_nodes.append(SummaryNode(node_id="corpus::summary", level=3, content=corpus_summary_text, parent_id=None,
                                  children_ids=[f"{doc['id']}::summary" for doc in documents], doc_id="corpus",
                                  embedding=embed_fn(corpus_summary_text)))
    return all_nodes
```

The build order is strictly bottom-up (chunks exist already, then sections, then documents, then the corpus), because each level's summarization input is the level directly below it — you cannot generate a document summary before its sections' summaries exist.

</details>

---

## Q5. How does query-time level routing work? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A cheap classifier model categorizes each incoming query into one of four scopes before any retrieval happens, and a fixed lookup table maps each category to the tree level(s) to search:

```python
ROUTER_PROMPT = """Classify this query's required abstraction level:
- "overview": broad topic/summary question. E.g. "What is this report about?"
- "section": targets a specific section/concept, no exact passage needed. E.g. "What does the methodology section say?"
- "chunk": needs a specific fact, figure, or verbatim detail. E.g. "What was Q3 revenue?"
- "multi": spans multiple levels (broad context + specific facts).
Output JSON: {"level": "overview"|"section"|"chunk"|"multi", "reasoning": "one sentence"}"""

LEVEL_MAP = {"overview": [2, 3], "section": [1], "chunk": [0], "multi": [0, 1, 2]}

def route_query(query: str) -> dict:
    resp = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=128,
                                   system=ROUTER_PROMPT, messages=[{"role": "user", "content": query}])
    return json.loads(resp.content[0].text)

def retrieve_from_tree(query, nodes_by_level, embed_fn, k=5) -> list:
    route = route_query(query)
    target_levels = LEVEL_MAP[route["level"]]
    query_emb = np.array(embed_fn(query))
    results = []
    for level in target_levels:
        sims = [(node, np.dot(query_emb, np.array(node.embedding)) /
                 (np.linalg.norm(query_emb) * np.linalg.norm(node.embedding) + 1e-9))
                for node in nodes_by_level.get(level, [])]
        sims.sort(key=lambda x: x[1], reverse=True)
        results.extend([node for node, _ in sims[:k]])
    return results
```

The router runs once per query on a cheap model (illustrative ~$0.00005/call on Haiku), which is negligible relative to the retrieval-quality benefit of not forcing every query through the same abstraction level — the alternative of always retrieving from Level 0 regardless of query scope is what this whole architecture exists to avoid.

</details>

---

## Q6. How is Recursive Document Summarization RAG different from RAPTOR? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| | RAPTOR (#13) | Recursive Summary RAG (#41) |
|---|---|---|
| Structure | Bottom-up clustering of similar chunks across documents | Top-down summarization within document boundaries |
| Hierarchy axis | Semantic similarity | Document structure (section → document → corpus) |
| Cross-document nodes | Yes — cluster nodes mix chunks from multiple documents | No — each node belongs to a single source document |
| Best for | Finding thematic connections across many documents | Navigating within-document structure at the right level |
| Summary content | Cluster topic summary | Faithful section/document summary |
| Retrieval for detail | Drill into cluster children | Drill into section chunks |

RAPTOR groups similar chunks bottom-up regardless of source document, which is powerful for thematic queries ("what do all these papers say about attention?") but loses document identity — a RAPTOR node might mix content from five different papers, making "which document says X?" unanswerable from the node alone. Recursive Summary RAG summarizes top-down within document boundaries, so every node preserves provenance — use it when users navigate documents individually (a legal contract, an annual report) and need to understand a single document at multiple granularities, rather than finding cross-document themes.

</details>

---

## Q7. When would a user want document-based navigation over topic-based navigation? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

```
Recursive Summary Tree                    RAPTOR
─────────────────────                    ──────
doc A                                    cluster_1
 ├── section A.1                          ├── chunk_A1 (doc A)
 │    ├── chunk_0                         ├── chunk_B3 (doc B)
 │    └── chunk_1                         └── chunk_C2 (doc C)
 └── section A.2                         cluster_2
      └── chunk_2                          ├── chunk_A5 (doc A)
                                           └── chunk_D1 (doc D)

Navigation: drill into a document         Navigation: drill into a topic
Best for: "What does doc A say about X?"  Best for: "What do all docs say about topic Y?"
```

Document-based navigation (this architecture) is the right fit when the unit of meaning to the user *is* the document — a specific contract, a specific filing, a specific research paper someone wants to understand in isolation, where "which document" is itself part of the answer. Topic-based navigation (RAPTOR) fits when the user's mental model is a topic that cuts across many documents, and which specific document any given fact came from is secondary to synthesizing the topic as a whole. A corpus of independent legal contracts (each self-contained, rarely compared against each other) favors this architecture; a corpus of research papers on a shared research question (where synthesizing across papers is the point) favors RAPTOR.

</details>

---

## Q8. How do you implement coarse-to-fine retrieval combining section-summary retrieval with chunk drill-down? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A powerful default pattern: always retrieve section summaries first (high precision, low noise), then drill down into the matched sections' actual chunks (high recall on the specific passage) — rather than relying on the router to correctly guess "chunk" vs. "section" up front:

```python
def drill_down(summary_node, nodes_by_id, query, embed_fn, k=3) -> list:
    """Given a retrieved section summary, fetch its most relevant child chunks."""
    child_nodes = [nodes_by_id[cid] for cid in summary_node.children_ids if cid in nodes_by_id]
    if not child_nodes:
        return [summary_node]
    query_emb = np.array(embed_fn(query))
    sims = [(node, np.dot(query_emb, np.array(node.embedding)) /
             (np.linalg.norm(query_emb) * np.linalg.norm(node.embedding) + 1e-9))
            for node in child_nodes]
    sims.sort(key=lambda x: x[1], reverse=True)
    return [node for node, _ in sims[:k]]

def coarse_to_fine_retrieve(query, nodes_by_level, nodes_by_id, embed_fn, top_sections=3, chunks_per_section=3) -> list:
    top_section_nodes = retrieve_from_tree(query, {1: nodes_by_level[1]}, embed_fn, k=top_sections)
    chunk_results = []
    for section_node in top_section_nodes:
        chunk_results.extend(drill_down(section_node, nodes_by_id, query, embed_fn, k=chunks_per_section))
    return chunk_results
```

```
Query: "What was the revenue growth rate in Q3?"
Step 1: find "Financial Results" section summary (score: 0.89)
Step 2: re-rank chunks within that section → "Q3 2023 revenue grew 18% YoY to $4.2B, driven by..."
```

This combination trades a small amount of extra latency (two retrieval passes instead of one) for eliminating an entire class of router error: a query that would have been misclassified as "section" when it actually needed exact figures still ends up at the right chunk, because drill-down always happens regardless of the initial classification's precision.

</details>

---

## Q9. How does retrieval from a specific tree level work under the hood? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Once the router (Q5) has selected which level(s) to search, retrieval within a level is ordinary cosine-similarity search over that level's node embeddings:

```python
def retrieve_from_tree(query, nodes_by_level, embed_fn, k=5) -> list:
    route = route_query(query)
    target_levels = LEVEL_MAP[route["level"]]
    query_emb = np.array(embed_fn(query))
    results = []
    for level in target_levels:
        candidates = nodes_by_level.get(level, [])
        sims = [(node, np.dot(query_emb, np.array(node.embedding)) /
                 (np.linalg.norm(query_emb) * np.linalg.norm(node.embedding) + 1e-9))
                for node in candidates]
        sims.sort(key=lambda x: x[1], reverse=True)
        results.extend([node for node, _ in sims[:k]])
    return results
```

The important structural detail is that `nodes_by_level` partitions the index by level *before* similarity search runs — a query routed to "section" never even computes similarity against chunk or document nodes, which both saves compute and, more importantly, avoids a section summary and a raw chunk ever competing directly for the same top-k slot despite being fundamentally different granularities of text that aren't meaningfully comparable by a single embedding-space distance.

</details>

---

## Q10. What are the key tuning knobs for this architecture, and how do you choose them? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Knob | Effect | Starting point |
|---|---|---|
| Chunks per section (tree fan-out at Level 0→1) | Fewer chunks per section summary preserves more detail per summary but produces more Level 1 nodes | Follows the document's natural section boundaries rather than an arbitrary fixed count |
| `top_sections` / `chunks_per_section` (coarse-to-fine, Q8) | Controls recall vs. context size trade-off for the default retrieval path | 3 sections × 3 chunks is a reasonable default; raise `top_sections` if section-level precision is weak |
| Router model choice and confidence handling | Determines routing accuracy and whether low-confidence routes should default to coarse-to-fine instead | A cheap model (Haiku-class) is sufficient given the router's output space is only four categories |
| Level 3 (corpus) update cadence | Since corpus summary regeneration is the most expensive step, it can lag behind Level 0-2 updates | Update on every document change for Levels 0-2; batch Level 3 updates on a schedule (Q16) |

The most consequential knob is actually a design decision, not a numeric parameter: whether to trust the router's classification directly (cheaper, one retrieval pass) or default to coarse-to-fine unconditionally (Q8's approach, slightly more expensive but immune to router misclassification, Q14). Teams with a well-calibrated router and cost-sensitive query volume lean toward direct routing; teams prioritizing recall over marginal cost lean toward always doing coarse-to-fine.

</details>

---

## Q11. How do you evaluate whether the level router is correctly classifying queries? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Build a labeled evaluation set of queries with their correct target level (overview/section/chunk/multi), ideally drawn from real user queries rather than synthetic ones, since real query phrasing is what the router will actually face in production. Measure:

1. **Routing accuracy** — does the router's predicted level match the labeled correct level? Break this down by category, since a router might be reliable on "chunk" queries (usually distinctively phrased around exact facts) but weak on distinguishing "overview" from "multi" (a genuinely blurrier boundary).
2. **Downstream answer quality conditioned on routing correctness** — for queries where the router got the level right vs. wrong, compare final answer quality; this quantifies how much a routing error actually costs in practice, which may be smaller than raw routing accuracy alone suggests if drill-down (Q8) is already partially compensating.
3. **Confusion matrix analysis** — which categories does the router confuse with which others? A router that frequently confuses "section" and "chunk" suggests the two categories' example queries in the prompt (Q5) need clearer boundary examples.

If routing accuracy is consistently below what the downstream quality can tolerate, the practical fix is often not a better router but switching the default retrieval path to unconditional coarse-to-fine (Q8), which sidesteps needing the router to be highly accurate on the section/chunk boundary specifically.

</details>

---

## Q12. How would you build a decision-gate benchmark for summary faithfulness across the tree? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Because every level above Level 0 is generated by summarizing the level below it, a hallucination or factual drift introduced at Level 1 propagates into Level 2 and then Level 3 — the corpus summary is, in effect, a summary of summaries of summaries, and errors compound rather than cancel out. A faithfulness gate needs to check each level against its *original source text*, not just against the level immediately below it:

```
1. Build a golden set: for a sample of documents, have a human (or a strong
   LLM-as-judge with the full source text available) verify each generated
   summary node (Level 1, 2, and 3) against the actual Level-0 source text
   it ultimately derives from -- not just its immediate parent summary,
   since an error at Level 1 that a Level 2 summary faithfully reflects is
   still a faithfulness failure traceable to Level 1's original generation.

2. Score each level's summaries on: factual precision (every claim in the
   summary traceable to the source) and factual recall (no major point from
   the source dropped) -- the same axes used for evaluation elsewhere in
   this bank, applied per tree level rather than to a single generation pass.

3. Gate: define a minimum faithfulness threshold per level (Level 1 summaries,
   being closest to source, should have the highest bar; Level 3's necessarily
   higher compression ratio may warrant a slightly more lenient precision
   threshold as long as it doesn't drop below a floor).

4. FAIL any tree rebuild where a sampled check of Level 1 summaries falls
   below threshold -- catching drift at the level closest to source is far
   cheaper than discovering it only after it has already propagated into
   Level 2 and Level 3 summaries built on top of the flawed Level 1 node.
```

The key principle: because compounding is the specific risk this architecture introduces (a flat RAG system has no equivalent "summary of a summary" propagation path), the faithfulness gate must check every level against ground truth independently, rather than assuming a passing check at Level 1 guarantees Level 2 and 3 built on it are also faithful.

</details>

---

## Q13. What is the "null-retrieval" failure mode when a query hits the corpus-level summary, and how do you defend against it? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

If a query is routed to Level 3 (the single corpus-wide summary), retrieval by definition returns exactly one node — a highly compressed, necessarily generic summary of the entire corpus — with no specifics to answer anything beyond the broadest orientation question. A query that genuinely needed more detail but got misrouted (or was ambiguous enough that "overview" was a defensible but ultimately unhelpful classification) produces an answer that sounds like a summary of everything and specifically like an answer to nothing.

**Defense:** treat Level 3 as a fallback tier that should almost always trigger a follow-up retrieval rather than terminating there — if the corpus summary is retrieved, automatically also retrieve the top Level 2 (document) summaries most similar to the query and include both in context, so the generator has both the broad framing and at least document-level specificity to draw from. This mirrors the coarse-to-fine pattern (Q8) applied specifically to the top of the tree: never let the single most-compressed node in the entire system be the sole basis for an answer, since by construction it has discarded the most information of any node in the tree.

</details>

---

## Q14. What happens when the level router misclassifies a query, and how do you debug or mitigate it? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A misclassification sends retrieval to the wrong tier entirely — a "chunk"-level query misrouted to "overview" retrieves only compressed summaries with no exact figures to extract, producing a vague or fabricated-sounding answer; an "overview" query misrouted to "chunk" retrieves a handful of specific passages that, individually, don't add up to a coherent broad answer, potentially causing the generator to over-generalize from an unrepresentative sample of chunks.

**Detection:** since misrouting doesn't throw an error (retrieval always returns *something*), it has to be caught via output-quality signals rather than a pipeline exception — track downstream answer quality (via user feedback, thumbs-up/down, or an automated groundedness check) segmented by the router's classification, and watch for a category whose downstream answers are systematically worse; this points at that category's routing being unreliable rather than a generation-quality problem generally.

**Mitigation ladder:** (1) tighten the router prompt's example queries for the confused category (Q11's confusion-matrix analysis tells you which pair to focus on); (2) for queries the router is genuinely uncertain about, have it emit a confidence score and fall back to coarse-to-fine (Q8) below a threshold, rather than committing to a single level on low confidence; (3) as a structural fix, default the whole system to always doing coarse-to-fine and use the router only as an optimization to skip levels when it's highly confident, rather than as a hard gate that determines correctness — this bounds the cost of a misclassification to "did unnecessary extra retrieval" rather than "returned an answer built from the wrong abstraction level entirely."

</details>

---

## Q15. What is the cost and latency profile of building and querying the summary tree at scale? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Stage | Operation | Cost per Document | Notes |
|---|---|---|---|
| Level 0 | Store raw chunks | $0 | Already done |
| Level 1 | Summarize N sections | ~$0.0002 × N | Haiku; cheap |
| Level 2 | Summarize document | ~$0.0002 | Haiku; 1 call |
| Level 3 | Summarize corpus | ~$0.002 | Sonnet for quality |
| Index build | Embed all summary nodes | ~$0.0001 × total_nodes | text-embedding-3-small |
| Query routing | Classify query level | ~$0.00005 | Haiku |
| Retrieval | ANN on multi-level index | <10ms per level | FAISS |

For an illustrative 100-document corpus with 10 sections/document and 5 chunks/section: total nodes = 100×5 chunks + 100×10 sections + 100 docs + 1 corpus = 1,601 nodes; total index-build cost is roughly $0.02 in LLM calls plus negligible embedding cost — the build cost is genuinely modest because every summarization call after Level 0 operates on already-compressed text (summaries of summaries), which keeps token counts, and therefore cost, low at every level above the base chunks.

Query-time cost is dominated by the router call (a single cheap classification) plus retrieval against whichever level(s) were selected — both trivial compared to the final generation call, meaning this architecture's total query-time cost overhead versus flat RAG is small: the router adds one cheap LLM call, and retrieval against a smaller, level-partitioned index is if anything faster than searching one large flat index of all chunks.

</details>

---

## Q16. How do you handle incremental updates when a document changes? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Only the modified document's subtree needs to be rebuilt, not the whole corpus:

1. Re-chunk the changed sections and re-embed the new chunks (Level 0).
2. Re-summarize only the affected sections (Level 1) — sections that didn't change don't need new summaries.
3. Re-summarize the document (Level 2), since at least one of its section summaries changed.
4. Optionally update the corpus summary (Level 3) if the document's contribution changed significantly enough to matter at that scale of compression.

Level 3 re-summarization is the most expensive step (it uses a stronger model for quality, per the cost table in Q15) and, because a single document's change rarely shifts what an entire corpus-wide summary should say, it can be deferred to a scheduled batch job (e.g., nightly or weekly) rather than triggered synchronously on every document change — the corpus summary tolerates being briefly stale in a way that a document's own summary should not, since users querying "what does document A say" expect that specific answer to reflect A's latest content immediately, while "what does this whole knowledge base cover broadly" tolerates a short lag.

Store a `last_modified` timestamp per node to efficiently detect which nodes are stale relative to their source: a node whose document hasn't changed since the node's own `last_modified` can be skipped entirely during a rebuild pass, which is what keeps incremental updates proportional to what actually changed rather than requiring a full corpus scan on every update.

</details>

---

## Q17. What security and trust risks does a multi-level, LLM-generated summary tree introduce? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

- **Hallucination amplification across levels** — a factual error introduced at Level 1 (a section summary subtly misstating a figure) becomes an *input* to Level 2's summarization, which has no way to distinguish "this claim is faithful to the original source" from "this claim was faithfully copied from a summary that was itself wrong." By the time an error reaches Level 3, it has been laundered through two additional rounds of LLM generation, each of which can further blur its connection back to the original source text — this is the compounding risk the faithfulness gate in Q12 exists specifically to catch.
- **Summary poisoning via source documents** — an adversarial document containing text specifically crafted to bias its own summary (analogous to prompt injection targeting the summarization step rather than a query-time generation step) can inject a false claim that then propagates upward through Level 2 and potentially Level 3, affecting how the *entire corpus* is characterized from a single poisoned source document — a broader blast radius than a poisoned document would have in a flat RAG system, where its influence is limited to being retrieved directly.
- **Provenance loss at higher levels** — even without malicious intent, a Level 3 corpus summary is, by construction, several summarization passes removed from any single source passage; a user or downstream system trusting a high-level summary as if it carried the same evidentiary weight as a directly-retrieved chunk is trusting something with meaningfully less traceable grounding, without any signal in the summary text itself indicating how many compression steps removed it is from the original source.

Mitigation: apply the same faithfulness auditing (Q12) as an ongoing production check, not just a one-time build gate; flag Level 2/3 nodes' outputs to users or downstream consumers as summaries-of-summaries rather than presenting them with the same confidence as a Level 0 chunk citation; and screen source documents for injection-style content before they're eligible for summarization, the same defense used against retrieval-time poisoning elsewhere in this bank.

</details>

---

## Q18. Design a Recursive Document Summarization RAG system for a large legal contract repository. `[Advanced]` `[Scenario]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Requirements:** users need to both understand an individual contract's structure (its own sections: parties, terms, obligations, termination clauses) and, separately, run cross-contract searches ("find all contracts with a specific termination-penalty structure").

```
1. Tree structure (this architecture): one summary subtree per contract,
   preserving contract identity at every level -- Level 0 chunks per
   clause, Level 1 summaries per contract section (parties, terms,
   obligations, termination), Level 2 whole-contract summary, Level 3
   summary across the full contract portfolio.

2. Within-contract navigation: "what does this contract's termination
   clause say?" routes to Level 1 (section) within that specific
   contract's subtree, then drills down (Q8) to the exact clause text
   for verbatim quoting -- legal use cases specifically require exact
   language, not paraphrase, so drill-down to Level 0 is mandatory
   whenever a specific clause is cited, never just the summary.

3. Cross-contract search (a gap this architecture alone doesn't fill):
   layer a SEPARATE flat or RAPTOR-style index over Level 0 chunks tagged
   by clause type (termination, indemnification, liability cap), so
   "find all contracts with X clause type" queries bypass the per-document
   tree entirely and search directly across all contracts' same-type
   clauses -- this is exactly the cross-document thematic search RAPTOR
   is suited for and this architecture's document-scoped hierarchy is not.

4. Faithfulness gating (Q12): mandatory for Level 1/2 contract summaries
   given the compliance stakes of a misrepresented clause; any summary
   node touching an obligation or penalty clause gets a stricter
   faithfulness threshold than a general "parties" section summary.

5. Incremental updates (Q16): contract amendments trigger re-summarization
   of only the amended section and the contract-level (Level 2) summary;
   the portfolio-level (Level 3) summary updates on a batch schedule.
```

The key design insight is recognizing that a single architecture doesn't have to solve both navigation patterns — this architecture excels at within-document structure, and the design explicitly layers a separate, purpose-built cross-document index alongside it rather than trying to force document-scoped summarization to also serve thematic cross-contract search, which Q6 and Q7 already establish it's structurally not well suited for.

</details>

---

## Q19. What is the research and practical origin of hierarchical document summarization for retrieval? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Recursive Document Summarization RAG doesn't trace to a single named paper the way DPR or RAPTOR do — it's the direct application of two well-established, separate ideas to RAG indexing: **hierarchical/multi-document summarization** research (a long-standing NLP subfield concerned with producing summaries at multiple levels of a document collection, predating LLM-based RAG entirely) and the general RAG pattern of **pre-computing retrievable artifacts offline** rather than doing all synthesis at query time, which RAPTOR (Beck et al., 2024) popularized specifically within the RAG context using clustering instead of document structure.

This architecture is best understood as RAPTOR's most direct structural sibling: same underlying motivation (give retrieval access to multiple abstraction levels, not just raw chunks) and much of the same offline-tree-building machinery, but organized along document structure instead of semantic clustering (Q2). Where RAPTOR has a specific originating paper to cite, this architecture is more accurately described as "the document-structure-preserving variant of RAPTOR's core idea," which is exactly the framing the comparison questions (Q6, Q7) are built around.

</details>

---

## Q20. What are the limitations of this architecture, and how might it evolve? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Current limitations:

- **No cross-document synthesis capability** (Q7, Q18) — the architecture's core strength (document-scoped provenance) is also its structural limitation; genuinely cross-document thematic queries need a separate index (as in Q18's design) rather than being answerable from this tree alone.
- **Compounding hallucination risk across levels** (Q12, Q17) — every level above Level 0 is generated text summarizing other generated or source text, and errors can propagate upward through multiple summarization passes with no built-in mechanism to catch drift before it compounds.
- **Router accuracy is a single point of failure for retrieval quality** (Q11, Q14) unless coarse-to-fine is used unconditionally, in which case the architecture pays a latency/cost tax on every query to avoid depending on router accuracy.
- **Document structure must actually be meaningful** — this architecture assumes documents have a natural section hierarchy worth preserving; unstructured or very short documents (a chat log, a single-paragraph memo) don't benefit from a 4-level tree the way a long structured report or contract does, and building one anyway adds overhead without a corresponding navigation benefit.

Likely evolution: hybrid designs that combine this architecture's document-preserving hierarchy with RAPTOR-style cross-document clustering as a second, complementary index over the same underlying chunks (rather than treating the two as mutually exclusive architectural choices, as Q18's design sketches for a legal-contract use case); and automated faithfulness scoring integrated directly into the tree-building pipeline (Q12) as a standard step, rather than a separate evaluation exercise run after the fact, so a Level 1 summary that fails a faithfulness check can trigger automatic regeneration before it ever propagates upward.

</details>

---

## Real-World Applications

| Application | Domain | Why This Architecture Fits |
|---|---|---|
| Individual contract or filing navigation | Legal / Compliance | Users need to understand one document's structure at multiple granularities, with provenance preserved |
| Long technical manual or specification Q&A | Engineering / Documentation | A manual's own section hierarchy is exactly the navigation structure users expect |
| Annual report / financial filing analysis | Finance | Analysts move between "what's the overall narrative" and "what's the exact Q3 figure" within one document |
| Single large research paper or book Q&A | Academia / Publishing | Preserves the document's own chapter/section structure rather than fragmenting it into a cross-document cluster |
| Enterprise knowledge base with independent, self-contained articles | Enterprise Ops | Each article's own structure matters more than finding cross-article themes |
