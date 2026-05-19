# 05 — Graph RAG

> Uses a knowledge graph for entity-aware retrieval, capturing relationships and enabling multi-hop reasoning.

---

## Q1. What is Graph RAG and what problem does it solve that vector RAG cannot? `[Basic]`

**Answer:**

**Graph RAG** replaces (or augments) the flat vector store with a **knowledge graph** — a structure of entities (nodes) and relationships (edges).

**Problem vector RAG cannot solve well:**
- *"What do the CEO of Company A and the CTO of Company B have in common?"*
- *"Which drugs interact with both Drug X and Drug Y?"*
- *"Trace the supply chain from raw material to final product."*

These require **multi-hop reasoning** — traversing relationships across multiple entities. Vector similarity finds relevant chunks but doesn't know that Entity A *acquired* Entity B or that Person X *reports to* Person Y. Graph RAG encodes this structure explicitly.

**Microsoft's GraphRAG** (2024) is the most notable production implementation, combining community detection with LLM-generated summaries of graph clusters.

---

## Q2. How is a knowledge graph constructed for Graph RAG? `[Intermediate]`

**Answer:**

Typical construction pipeline:

1. **Entity extraction** — Use an LLM or NER model to extract entities (people, orgs, locations, concepts) from documents.
2. **Relationship extraction** — Extract typed relationships between entities (`CEO_OF`, `ACQUIRED`, `LOCATED_IN`).
3. **Entity resolution** — Deduplicate entities that refer to the same real-world object ("Apple Inc." vs "Apple").
4. **Graph storage** — Store in a graph database (Neo4j, Amazon Neptune, TigerGraph) or in-memory (NetworkX for smaller graphs).
5. **Embedding nodes** — Optionally embed node descriptions for hybrid graph + vector retrieval.

This pipeline is expensive to build and maintain but pays off for corpora with dense relational structure (legal, biomedical, financial).

---

## Q3. What is community detection in Microsoft's GraphRAG and why is it important? `[Intermediate]`

**Answer:**

Microsoft's GraphRAG applies **Leiden community detection** to the knowledge graph to identify clusters of closely related entities (communities).

For each community, an LLM generates a **community summary** — a paragraph describing the key entities, relationships, and themes within that cluster.

**Why it matters:**

- Enables **global queries** — questions that require synthesizing information across many documents (e.g., "What are the main themes in this corpus?"). Pure vector RAG cannot answer these because no single chunk contains the answer.
- Provides **multi-level summarization** — community summaries can be hierarchical (sub-communities → communities → top-level).
- At query time, community summaries are retrieved like documents, but they represent synthesized knowledge rather than raw text chunks.

---

## Q4. How does Graph RAG handle multi-hop queries? `[Advanced]`

**Answer:**

For a query like *"Who funded the company that acquired OpenAI's main competitor?"*:

1. **Entity extraction** — Identify "OpenAI's main competitor" → resolve to Anthropic.
2. **Graph traversal** — Find `ACQUIRED_BY` edges from Anthropic → find the acquiring company.
3. **Hop 2** — Find `FUNDED_BY` edges from that company.
4. **Retrieve context** — Pull document chunks associated with the entities and relationships found.
5. **Generate** — Pass the traversal result + supporting chunks to the LLM.

**Implementation approaches:**
- **Beam search** — Explore top-k most relevant paths from the starting entity.
- **Subgraph extraction** — Extract the subgraph around relevant entities within N hops.
- **G-Retriever** — A specialized model that jointly reasons over graph structure and text.

---

## Q5. What are the trade-offs of Graph RAG vs. vector RAG in a production system? `[Advanced]`

**Answer:**

| Dimension | Vector RAG | Graph RAG |
|---|---|---|
| **Setup complexity** | Low | High (entity extraction, graph construction) |
| **Query types** | Semantic similarity | Relational, multi-hop |
| **Maintenance** | Re-embed on doc changes | Re-extract entities + update graph |
| **Latency** | Low (single ANN query) | Higher (graph traversal) |
| **Global queries** | Poor | Excellent (community summaries) |
| **Hallucination risk** | Medium | Lower (structured facts) |
| **Best for** | FAQs, document search | Research corpora, enterprise knowledge |

**Recommendation:** Use vector RAG as the default; add Graph RAG when your queries are inherently relational or when users ask "big picture" synthesis questions across a large corpus.
