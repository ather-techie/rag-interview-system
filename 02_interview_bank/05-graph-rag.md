# 05 — Graph RAG

> Uses a knowledge graph for entity-aware retrieval, capturing relationships and enabling multi-hop reasoning.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
Docs
  │
  ▼
Entity Extraction (NER / LLM) ──► Relationship Extraction (typed edges)
  │
  ▼
Entity Resolution (dedupe "Apple" vs "Apple Inc.")
  │
  ▼
Graph Store  ──────────────────────┐
  │                                │
  ▼                                │
Community Detection (Leiden)       │
  + LLM Community Summarization    │
  │                                │
  ▼                                ▼
       Hybrid Graph + Vector Retrieval
       (graph traversal narrows candidates,
        vector search enriches with text context)
  │
  ▼
Generator LLM → Answer
```

### Key Components

| Component | Responsibility |
|---|---|
| Entity Extractor | Identifies people, orgs, locations, and concepts from documents (NER/LLM) |
| Relationship Extractor | Extracts typed edges between entities (`CEO_OF`, `ACQUIRED`, `LOCATED_IN`) |
| Entity Resolver | Deduplicates mentions referring to the same real-world entity |
| Graph Database | Stores nodes/edges and serves Cypher-style traversal queries |
| Community Detector | Clusters related entities (Leiden) and generates LLM summaries for global/synthesis queries |
| Hybrid Retriever | Combines graph traversal (structural facts) with vector search (semantic enrichment) |
| Generator | Synthesizes an answer from graph facts + retrieved supporting text |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Graph Databases | Neo4j, NebulaGraph, Amazon Neptune |
| End-to-End Framework | Microsoft GraphRAG |
| Entity Extraction (NER) | spaCy, GLiNER |
| In-Memory Graphs | NetworkX |

---

## Q1. What is Graph RAG and what problem does it solve that vector RAG cannot? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Graph RAG** replaces (or augments) the flat vector store with a **knowledge graph** — a structure of entities (nodes) and relationships (edges).

**Problem vector RAG cannot solve well:**
- *"What do the CEO of Company A and the CTO of Company B have in common?"*
- *"Which drugs interact with both Drug X and Drug Y?"*
- *"Trace the supply chain from raw material to final product."*

These require **multi-hop reasoning** — traversing relationships across multiple entities. Vector similarity finds relevant chunks but doesn't know that Entity A *acquired* Entity B or that Person X *reports to* Person Y. Graph RAG encodes this structure explicitly.

**Microsoft's GraphRAG** (2024) is the most notable production implementation, combining community detection with LLM-generated summaries of graph clusters.

```
[Person] Alice ──CEO_OF──► [Company] Acme Corp
                                  │
                             ACQUIRED
                                  │
                                  ▼
[Company] BetaCorp ◄──FOUNDED_BY── [Person] Bob
                │
          LOCATED_IN
                │
                ▼
       [Location] San Francisco
```

</details>

---

## Q2. How is a knowledge graph constructed for Graph RAG? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Typical construction pipeline:

1. **Entity extraction** — Use an LLM or NER model to extract entities (people, orgs, locations, concepts) from documents.
2. **Relationship extraction** — Extract typed relationships between entities (`CEO_OF`, `ACQUIRED`, `LOCATED_IN`).
3. **Entity resolution** — Deduplicate entities that refer to the same real-world object ("Apple Inc." vs "Apple").
4. **Graph storage** — Store in a graph database (Neo4j, Amazon Neptune, TigerGraph) or in-memory (NetworkX for smaller graphs).
5. **Embedding nodes** — Optionally embed node descriptions for hybrid graph + vector retrieval.

This pipeline is expensive to build and maintain but pays off for corpora with dense relational structure (legal, biomedical, financial).

</details>

---

## Q3. What is community detection in Microsoft's GraphRAG and why is it important? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Microsoft's GraphRAG applies **Leiden community detection** to the knowledge graph to identify clusters of closely related entities (communities).

For each community, an LLM generates a **community summary** — a paragraph describing the key entities, relationships, and themes within that cluster.

**Why it matters:**

- Enables **global queries** — questions that require synthesizing information across many documents (e.g., "What are the main themes in this corpus?"). Pure vector RAG cannot answer these because no single chunk contains the answer.
- Provides **multi-level summarization** — community summaries can be hierarchical (sub-communities → communities → top-level).
- At query time, community summaries are retrieved like documents, but they represent synthesized knowledge rather than raw text chunks.

</details>

---

## Q4. How does Graph RAG handle multi-hop queries? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

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

```python
import networkx as nx

# Build graph from extracted triples
G = nx.DiGraph()
G.add_edge("Anthropic", "Google", relation="FUNDED_BY")
G.add_edge("Anthropic", "Claude", relation="CREATED")

# Multi-hop: find all entities reachable within 2 hops from a seed
seed = "Anthropic"
subgraph_nodes = nx.single_source_shortest_path_length(G, seed, cutoff=2)
relevant_entities = list(subgraph_nodes.keys())
# Then retrieve document chunks associated with relevant_entities
```

</details>

---

## Q5. What are the trade-offs of Graph RAG vs. vector RAG in a production system? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

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

</details>

---

## Q6. How do you combine a knowledge graph with a vector store for hybrid Graph+Vector retrieval? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Hybrid Graph+Vector retrieval uses the graph to identify relevant entities, then falls back to vector search for richer context.

```
Query: "What companies has the CEO of TechCorp invested in?"

Stage 1: Entity extraction
  → Identify "TechCorp", "CEO"

Stage 2: Graph lookup (structural)
  → CEO_OF(TechCorp) → Alice
  → Find all INVESTED_IN edges from Alice
  → → Candidates: StartupA, StartupB, StartupC

Stage 3: Vector search (semantic enrichment)
  For each candidate, search: "Alice investment in {Candidate}"
  → Retrieve rich context chunks

Stage 4: Synthesis
  Merge graph facts + vector context → Answer
```

**Implementation:**

```python
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Chroma
import networkx as nx

class HybridGraphVectorRetriever:
    def __init__(self, graph_db, vectorstore):
        self.graph = graph_db
        self.vectorstore = vectorstore
    
    def retrieve(self, query: str, k: int = 5):
        # Step 1: Extract entities from query
        entities = extract_entities(query)  # ["TechCorp", "CEO"]
        
        # Step 2: Graph-based retrieval
        graph_results = []
        for entity in entities:
            # Cypher: Find neighbors of this entity
            neighbors = self.graph.query(f"""
                MATCH (n {{name: '{entity}'}})-->(m)
                RETURN m.name, m.type
                LIMIT 10
            """)
            graph_results.extend(neighbors)
        
        # Step 3: Vector search for each graph result
        vector_results = []
        for result in graph_results:
            entity_name = result["m.name"]
            # Semantic search around this entity
            vector_context = self.vectorstore.similarity_search(
                f"{query} {entity_name}", k=3
            )
            vector_results.extend(vector_context)
        
        # Step 4: Deduplicate and merge
        seen = set()
        merged = []
        for result in vector_results:
            doc_id = result.metadata.get("doc_id")
            if doc_id not in seen:
                seen.add(doc_id)
                merged.append(result)
        
        return merged[:k]
```

**Advantages:**
- Combines structural precision (graph) with semantic richness (vectors).
- Reduces hallucinations (graph provides facts).
- Faster than pure graph traversal (graph narrows the search space, then vectors find nuances).

</details>

---

## Q7. How do you query a Neo4j knowledge graph in a Graph RAG pipeline? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Neo4j queries use Cypher, a graph query language. Here's how to integrate it into a RAG pipeline:

```python
from neo4j import GraphDatabase

class Neo4jGraphRAG:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def extract_query_entities(self, query: str) -> list[str]:
        """Extract entities from user query (e.g., using NER)."""
        # Pseudo-code; use spaCy, transformers, or LLM
        entities = extract_entities_with_nlp(query)
        return entities
    
    def multi_hop_traversal(self, start_entity: str, hops: int = 2) -> dict:
        """Traverse the graph N hops from a starting entity."""
        with self.driver.session() as session:
            cypher_query = f"""
                MATCH path = (start:{start_entity})-[*1..{hops}]-(neighbor)
                RETURN path, neighbor.name as neighbor_name, 
                       [rel in relationships(path) | type(rel)] as relation_types
                LIMIT 20
            """
            result = session.run(cypher_query)
            paths = [record for record in result]
            return paths
    
    def retrieve_related_documents(self, entity_name: str) -> list[dict]:
        """Find documents mentioning a given entity."""
        with self.driver.session() as session:
            query = f"""
                MATCH (e:Entity {{name: '{entity_name}'}})-[:MENTIONED_IN]->(doc:Document)
                RETURN doc.id, doc.content, e.type as entity_type
                LIMIT 5
            """
            result = session.run(query)
            docs = [record for record in result]
            return docs
    
    def rag_query(self, user_query: str) -> str:
        """End-to-end Graph RAG."""
        # Step 1: Extract entities
        entities = self.extract_query_entities(user_query)
        
        # Step 2: Multi-hop traversal from each entity
        all_neighbors = []
        for entity in entities:
            paths = self.multi_hop_traversal(entity, hops=2)
            for path in paths:
                all_neighbors.append(path["neighbor_name"])
        
        # Step 3: Retrieve documents for each entity found
        all_docs = []
        for neighbor in set(all_neighbors):
            docs = self.retrieve_related_documents(neighbor)
            all_docs.extend(docs)
        
        # Step 4: Generate answer with LLM
        context = "\n".join([f"- {doc['content']}" for doc in all_docs])
        prompt = f"""Given the context below, answer: {user_query}
        
Context:
{context}

Answer:"""
        
        answer = llm.invoke(prompt)
        return answer

# Cypher query examples
cypher_examples = {
    "Find all investments by a person": """
        MATCH (p:Person {name: 'Alice'})-[inv:INVESTED_IN]->(c:Company)
        RETURN c.name, inv.amount, inv.date
    """,
    "Find common board members": """
        MATCH (p1:Person)-[:BOARD_MEMBER_OF]->(c1:Company),
              (p2:Person)-[:BOARD_MEMBER_OF]->(c2:Company)
        WHERE c1.id = {company_id_1} AND c2.id = {company_id_2}
        RETURN p1, p2
    """,
    "Supply chain tracing": """
        MATCH path = (raw:Material)-[:USED_IN*]->(final:Product)
        WHERE raw.name = 'Lithium' AND final.name = 'Tesla Battery'
        RETURN path
    """
}
```

**Key Cypher patterns:**
- `-[*1..N]→` traverses 1 to N hops.
- `relationships(path)` extracts all relationship types.
- `LIMIT` prevents unbounded result sets.

</details>

---

## Q8. How does PathRAG differ from Microsoft's GraphRAG community-detection approach? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Both use graphs for RAG, but their indexing and query strategies differ significantly:

| Aspect | Microsoft GraphRAG | PathRAG |
|--------|-------------------|---------|
| **Indexing** | Community detection (Leiden) + LLM summarization of communities | Dense paths through the graph |
| **Query type** | Local + global (community summaries answer big-picture Qs) | Path-based (answer Qs via graph traversal) |
| **Retrieval mechanism** | Rank communities by relevance; retrieve community summary + member chunks | Beam search to find shortest/most relevant paths |
| **Strengths** | Excellent for synthesis across many docs; scalable to large graphs | Efficient multi-hop traversal; precise fact retrieval |
| **Weaknesses** | Expensive LLM-based summarization at indexing time | Less good for global/summary queries |

**Example query: "What supply chain disruptions affected automotive?" **

**GraphRAG approach:**
1. Community detection finds clusters (e.g., "Semiconductor Supply Chain", "Logistics Companies")
2. Retrieves community summaries mentioning "disruption"
3. Synthesis → Answer

**PathRAG approach:**
1. Identify entities related to "automotive" and "supply chain"
2. Beam search finds shortest paths connecting them through disruption-related nodes
3. Traverse and collect path evidence → Answer

```python
# PathRAG example: Beam search for paths
class PathRAGRetriever:
    def __init__(self, graph):
        self.graph = graph
    
    def beam_search_paths(self, start_entity: str, goal_entity: str, 
                         beam_width: int = 5, max_hops: int = 3) -> list[list]:
        """Find top-k shortest paths via beam search."""
        import heapq
        
        # Priority queue: (cost, path)
        queue = [(0, [start_entity])]
        visited = set()
        paths = []
        
        while queue and len(paths) < beam_width:
            cost, path = heapq.heappop(queue)
            
            if len(path) > max_hops:
                continue
            
            last_entity = path[-1]
            if last_entity == goal_entity:
                paths.append(path)
                continue
            
            # Expand neighbors
            neighbors = self.graph.neighbors(last_entity)
            for neighbor in neighbors:
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    # Cost = path length + relevance score
                    relevance = self.compute_relevance(neighbor, goal_entity)
                    new_cost = len(new_path) - relevance
                    heapq.heappush(queue, (new_cost, new_path))
                    visited.add(neighbor)
        
        return paths

# Recommendation: Use GraphRAG for summary/synthesis questions, PathRAG for fact retrieval.
```

</details>

---

## Q9. What entity resolution techniques prevent graph fragmentation? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Entity resolution (linking mentions to canonical entities) is critical — without it, "Alice Smith", "A. Smith", "Alice M. Smith" create separate nodes instead of one.

**Problem:**
```
Alice ┐
A. Smith ├─→ Should be: [Alice Smith] (one entity)
Alice M. Smith ┘
```

**Techniques:**

| Technique | Method | Pros | Cons |
|-----------|--------|------|------|
| **Exact string match** | "Alice Smith" == "Alice Smith" | Fast, deterministic | Misses variations |
| **Fuzzy matching** | Edit distance <threshold | Handles typos | Slow at scale |
| **Embedding similarity** | Embed mentions, cluster by cosine sim | Captures semantics | Hyperparameter tuning |
| **LLM-based linking** | LLM judges if two mentions refer to same entity | High accuracy | Expensive |
| **Blocking + matching** | First block candidates (same last name), then match | Scalable | Multi-stage complexity |

**Production pipeline:**

```python
import difflib
from sentence_transformers import SentenceTransformer

class EntityResolver:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.entity_map = {}  # mention → canonical_id
        self.canonical_entities = {}  # id → {name, type, aliases}
    
    def resolve(self, mention: str, entity_type: str) -> str:
        """Resolve a mention to a canonical entity ID."""
        
        # Stage 1: Exact match
        if mention in self.entity_map:
            return self.entity_map[mention]
        
        # Stage 2: Fuzzy match within type
        candidates = [
            (eid, name) 
            for eid, entity in self.canonical_entities.items()
            if entity["type"] == entity_type
        ]
        
        best_match = None
        best_score = 0.8  # Threshold
        
        for eid, canonical_name in candidates:
            # Fuzzy similarity
            ratio = difflib.SequenceMatcher(None, mention.lower(), 
                                          canonical_name.lower()).ratio()
            if ratio > best_score:
                best_score = ratio
                best_match = eid
        
        if best_match:
            self.entity_map[mention] = best_match
            return best_match
        
        # Stage 3: Embedding-based similarity
        mention_emb = self.embedding_model.encode(mention)
        for eid, entity in self.canonical_entities.items():
            canonical_emb = self.embedding_model.encode(entity["name"])
            sim = np.dot(mention_emb, canonical_emb)  # Cosine
            if sim > 0.85:  # High threshold for embedding
                self.entity_map[mention] = eid
                return eid
        
        # Stage 4: Create new entity
        new_id = self.create_entity(mention, entity_type)
        self.entity_map[mention] = new_id
        return new_id
    
    def create_entity(self, name: str, entity_type: str) -> str:
        new_id = f"{entity_type}_{len(self.canonical_entities)}"
        self.canonical_entities[new_id] = {
            "name": name,
            "type": entity_type,
            "aliases": [name]
        }
        return new_id

# Evaluation: Graph fragmentation metric
def measure_fragmentation(graph, gold_standard_clusters):
    """How many entities are wrongly split (false positives)?"""
    actual_clusters = find_connected_components(graph)
    fragmentation_errors = len(actual_clusters) - len(gold_standard_clusters)
    return max(fragmentation_errors, 0)
```

</details>

---

## Q10. How do you evaluate Graph RAG quality beyond standard vector-RAG metrics? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Graph RAG requires specialized metrics that assess structural and relational correctness.

| Metric | Definition | Example |
|--------|-----------|---------|
| **Graph Coverage** | % of entities in docs that are in the graph | If 100 entities in docs, graph has 95 → 95% coverage |
| **Relation F1** | Precision & recall of extracted relationships | Gold: 50 relations, extracted: 45 correct, 10 wrong → F1 ≈ 0.90 |
| **Hop Accuracy** | % of multi-hop paths that are factually correct | User query requires 2-hop path; system finds correct path |
| **Community Recall** | Can retrieved community summaries answer synthesis questions? | Query: "main themes?"; community summary covers all themes |
| **Factuality** | % of graph-derived facts verified against source docs | System states "A acquired B"; document confirms? |

**Evaluation pipeline:**

```python
from collections import defaultdict
import numpy as np

class GraphRAGEvaluator:
    def __init__(self, gold_graph, gold_summaries):
        self.gold_graph = gold_graph  # Ground truth knowledge graph
        self.gold_summaries = gold_summaries  # Annotated community summaries
        self.test_queries = []  # Queries requiring graph traversal
    
    def evaluate_graph_coverage(self, extracted_graph):
        """Measure % of entities correctly extracted."""
        gold_entities = set(self.gold_graph.nodes())
        extracted_entities = set(extracted_graph.nodes())
        
        coverage = len(gold_entities & extracted_entities) / len(gold_entities)
        return coverage
    
    def evaluate_relation_f1(self, extracted_graph):
        """Measure precision & recall of relationships."""
        gold_edges = set(self.gold_graph.edges())
        extracted_edges = set(extracted_graph.edges())
        
        tp = len(gold_edges & extracted_edges)
        fp = len(extracted_edges - gold_edges)
        fn = len(gold_edges - extracted_edges)
        
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        return {"precision": precision, "recall": recall, "f1": f1}
    
    def evaluate_hop_accuracy(self, system, queries):
        """% of multi-hop queries answered correctly."""
        correct = 0
        for query, expected_path in queries:
            actual_path = system.beam_search(query)
            if actual_path == expected_path:
                correct += 1
        
        return correct / len(queries)
    
    def evaluate_community_recall(self, system, community_queries):
        """% of synthesis questions answered by community summaries."""
        correct = 0
        for query, expected_themes in community_queries:
            # Retrieve community summaries
            communities = system.retrieve_communities(query)
            retrieved_summaries = [c["summary"] for c in communities]
            merged_summary = " ".join(retrieved_summaries)
            
            # Check if all expected themes are covered
            all_present = all(theme in merged_summary for theme in expected_themes)
            if all_present:
                correct += 1
        
        return correct / len(community_queries)
    
    def run_full_evaluation(self, extracted_graph, system) -> dict:
        coverage = self.evaluate_graph_coverage(extracted_graph)
        relation_metrics = self.evaluate_relation_f1(extracted_graph)
        hop_acc = self.evaluate_hop_accuracy(system, self.test_queries[:20])
        comm_recall = self.evaluate_community_recall(system, self.test_queries[20:])
        
        overall_score = (
            coverage * 0.25 +
            relation_metrics["f1"] * 0.25 +
            hop_acc * 0.25 +
            comm_recall * 0.25
        )
        
        return {
            "graph_coverage": coverage,
            "relation_f1": relation_metrics["f1"],
            "hop_accuracy": hop_acc,
            "community_recall": comm_recall,
            "overall_score": overall_score
        }

# Example thresholds:
# - Graph coverage > 90%
# - Relation F1 > 0.85
# - Hop accuracy > 80%
# - Community recall > 85%
```

**Production evaluation cadence:**
- Weekly: automated metrics (coverage, F1) on held-out test set.
- Monthly: manual annotation of 100 queries (hop accuracy, factuality).
- Quarterly: user feedback on synthesis questions (community recall).
- Post-deployment: monitor false positive relations (extra validation).


</details>

---

## Q11. How do you estimate and control the cost of knowledge graph construction and graph traversal at production scale? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Knowledge graph construction cost:**

Building a graph from raw documents involves multiple expensive steps:

| Step | LLM Calls | Latency | Cost/1M docs |
|------|-----------|---------|--------------|
| **Entity extraction** | 1 per doc | 200ms | $20 (0.02/doc) |
| **Relationship extraction** | 1 per doc | 300ms | $20 |
| **Entity linking (deduplication)** | 0.1 per unique entity | 50ms | $5 |
| **Graph validation** | 1 per 100 edges | 100ms | $2 |
| **Total construction** | — | — | **$47** |

For a 100M document corpus (typical enterprise), initial construction cost: $4,700.

**Cost optimization strategies:**

1. **Batch entity/relation extraction** — Process documents in batches (100–1000 at a time):
   ```python
   # Sequential: 1M docs × 2 LLM calls = 2M LLM calls
   for doc in docs:
       entities = extract_entities_llm(doc)
       relations = extract_relations_llm(doc)
   
   # Batched: 1M / batch_size batches × 2 = 20K LLM calls (100x cheaper)
   batch_size = 50_000
   for batch in chunks(docs, batch_size):
       entities_batch = batch_extract_entities(batch)
       relations_batch = batch_extract_relations(batch)
   ```
   - Savings: 100x cost reduction, minimal quality loss.

2. **Rule-based extraction + LLM refinement** — Use regex/heuristics first, only use LLM for ambiguous cases:
   ```python
   entities = extract_entities_regex(doc)  # Free
   
   # Only ~10% need LLM refinement
   ambiguous_entities = filter(lambda e: e.confidence < 0.7, entities)
   refined = llm_refine_entities(ambiguous_entities)
   
   # Save 90% of LLM calls
   ```

3. **Lazy graph construction** — Only build graph for queried subdomains:
   - Don't build the entire graph upfront.
   - When a user queries a domain (e.g., "customer relationships"), construct graph on-demand.
   - Savings: 50–80% of construction cost (only build 20–50% of graph).

4. **Graph traversal caching** — Cache frequently traversed paths:
   ```python
   traversal_cache = {}
   
   def traverse_path_cached(start_entity, target_relation, depth=2):
       cache_key = (start_entity, target_relation, depth)
       if cache_key in traversal_cache:
           return traversal_cache[cache_key]  # Free
       
       path = bfs_graph(start_entity, target_relation, depth)
       traversal_cache[cache_key] = path
       return path
   ```
   - 30–50% cache hit rate → 30–50% traversal cost reduction.

5. **Approximate graph algorithms** — Use sampling for large graphs:
   ```python
   # Full BFS on 1M nodes: expensive
   results_full = bfs_exact(graph, start, relation)
   
   # Sample 10K random nodes, search in subgraph: fast
   sample_nodes = random.sample(graph.nodes(), 10_000)
   subgraph = graph.subgraph(sample_nodes)
   results_sampled = bfs_approximate(subgraph, start, relation)
   
   # Trade-off: 5–10% accuracy loss, 100x speedup
   ```

**Example cost structure at scale (100M docs):**

```
Initial construction (one-time):
  - Entity extraction: 100M × $0.0001 = $10K
  - Relation extraction: 100M × $0.0001 = $10K
  - Entity linking (10M unique): 10M × $0.0005 = $5K
  - Total: $25K (amortized per user: $0.0003/query if 100M queries/month)

Per-query traversal:
  - Average path length: 3–5 hops
  - Per-hop cost: $0.001 (if using LLM for reasoning)
  - Per-query cost: $0.005
  - Monthly: 100M queries × $0.005 = $500K (ouch!)

With optimizations:
  - Caching (50% hit rate): $0.003/query
  - Approximate algorithms: $0.001/query
  - Total: $0.002/query → $200K/month (60% savings)
```

**Monitoring and cost control:**

- Track graph construction cost per corpus update (daily/weekly).
- Monitor traversal latency; flag slow queries (potential runaway).
- Set budget alerts (e.g., "$500/day max spending on graph operations").

</details>

---

## Q12. How can adversaries poison a knowledge graph through entity spoofing or false relationship injection, and what integrity controls prevent this? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Attack 1: Entity spoofing**

An attacker injects a fake entity that impersonates a real one:

```
Real entity: "Apple Inc." (NASDAQ: AAPL)
Fake entity injected: "Apple Inc." (NASDAQ: APPL) ← typo, different company

Graph now has both entities with similar names.
User queries "What is Apple's stock price?" → ambiguous, might retrieve wrong entity.
```

**Attack 2: False relationship injection**

Attacker injects a fake relationship between entities:

```
Real: "John Smith (CEO) → works_at → Company X"
Fake: "John Smith (CEO) → works_at → Competitor Y" ← injected

Graph now has conflicting relationships.
User queries "Where does John Smith work?" → returns both answers.
System confidence degrades due to contradiction.
```

**Attack 3: Trojan relationships**

Attacker creates relationships that lead queries to unexpected results:

```
Benign query: "What products does Company X make?"
Attacker injects: Company X → supplies_to → Malicious Competitor Z
Result: User gets info about Company Z instead of X's products.
```

**Defences:**

**1. Entity disambiguation and canonical naming:**

Maintain a canonical entity registry with strict rules:

```python
CANONICAL_ENTITIES = {
    "Apple Inc. NASDAQ:AAPL": {
        "aliases": ["Apple", "AAPL"],
        "verified": True,
        "source": "official_sec_filing"
    },
    "Apple Inc. private": {
        "aliases": ["Apple Company"],
        "verified": False,
        "source": "unverified_web_scrape"
    }
}

def resolve_entity(entity_name):
    for canonical, info in CANONICAL_ENTITIES.items():
        if entity_name in info["aliases"]:
            if info["verified"]:
                return canonical  # Return canonical form
            else:
                return None  # Flag as unverified
```

**2. Relationship validation via multiple sources:**

Only accept relationships that are corroborated by multiple independent sources:

```python
def add_relationship_with_validation(source_entity, relation, target_entity, source_doc):
    # Check if relationship exists from other sources
    existing_sources = relationship_sources.get((source_entity, relation, target_entity), [])
    
    # Only add if:
    # - Multiple independent sources agree, OR
    # - Source is high-trust (e.g., official database)
    if len(existing_sources) >= 2 or is_high_trust_source(source_doc):
        graph.add_edge(source_entity, target_entity, relation=relation)
    else:
        # Mark as unverified/pending
        pending_relationships.add((source_entity, relation, target_entity))
```

**3. Source attribution and trust scoring:**

Track where each entity and relationship came from:

```python
class GraphEntity:
    def __init__(self, name, source_docs, trust_score=0.5):
        self.name = name
        self.sources = source_docs
        self.trust_score = trust_score  # 0-1
        self.is_verified = any(is_high_trust(doc) for doc in source_docs)
    
    def add_source(self, doc):
        self.sources.append(doc)
        # Update trust score based on new source credibility
        self.trust_score = mean([credibility(doc) for doc in self.sources])

# During traversal, weight by trust score
def traverse_with_trust_weighting(start_entity, relation, depth=2):
    paths = bfs_graph(graph, start_entity, relation, depth)
    
    # Filter low-trust paths
    trusted_paths = [
        path for path in paths
        if all(entity.trust_score > 0.6 for entity in path)
    ]
    
    return trusted_paths
```

**4. Contradiction detection:**

Flag relationships that contradict each other:

```python
def detect_contradictions():
    for (source, relation, target) in graph.edges():
        entity = graph.nodes()[source]
        
        # If entity has multiple conflicting relationships, flag
        alternate_targets = graph.neighbors(source, relation=relation)
        
        if len(alternate_targets) > 1:
            # Multiple targets for same (entity, relation) pair
            trust_scores = [graph.nodes()[t].trust_score for t in alternate_targets]
            
            if max(trust_scores) - min(trust_scores) > 0.3:
                # Large difference in trust → contradiction
                log_contradiction(source, relation, alternate_targets)
                flag_for_review()
```

**5. Graph access control and auditability:**

Enforce authorization and log all graph modifications:

```python
class AuditedKnowledgeGraph:
    def add_entity(self, entity, source_doc, user_id):
        if not has_permission(user_id, "write_entity", entity.type):
            raise PermissionError(f"User {user_id} cannot write entities")
        
        # Log the modification
        self.audit_log.append({
            "timestamp": datetime.now(),
            "user_id": user_id,
            "action": "add_entity",
            "entity": entity,
            "source": source_doc
        })
        
        super().add_entity(entity, source_doc)
```

**6. Periodic integrity audits:**

Regularly validate the graph against trusted sources:

```python
def audit_graph_integrity():
    # Sample entities and relationships
    sample_entities = random.sample(graph.nodes(), 100)
    
    for entity in sample_entities:
        # Cross-check against external authoritative source (e.g., Wikidata)
        external_truth = wikidata.lookup(entity.name)
        
        if external_truth is None:
            # Entity doesn't exist in external source
            log_suspicious_entity(entity)
        
        # Verify relationships
        for relation, neighbors in graph[entity].items():
            if not external_truth.has_relation(relation, neighbors):
                log_suspicious_relationship(entity, relation, neighbors)
```

**7. Graph rollback and versioning:**

Maintain historical versions to recover from poisoning:

```python
class VersionedKnowledgeGraph:
    def __init__(self):
        self.versions = {}
        self.current_version = 0
    
    def snapshot(self):
        # Create immutable snapshot of current graph
        self.versions[self.current_version] = deepcopy(self.graph)
        self.current_version += 1
    
    def rollback(self, version):
        # Revert to previous version if poisoning is detected
        if self.detect_anomalies():
            self.graph = deepcopy(self.versions[version - 1])
            log_rollback(version)
```

**Defence-in-depth:**

1. Entity canonicalization and strict naming rules.
2. Multiple-source validation for relationships.
3. Source attribution and trust scoring.
4. Contradiction detection.
5. Access control and auditability.
6. Periodic integrity audits against external sources.
7. Versioning and rollback capabilities.

Combined, these layers make graph poisoning expensive and detectable.

</details>

---

## Q13. Walk through the Graph RAG architecture end-to-end. `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

```
Docs → LLM Entity/Relationship Extraction → Knowledge Graph
                                                    │
                                          Community Detection (Leiden)
                                                    │
                                          LLM Community Summarization
                                                    │
Query → Route to local (entity-anchored) or global (community-summary) search
                                                    │
                                          Retrieved graph context → Generator → Answer
```

Every stage after raw documents is an LLM call working over the previous stage's output — extraction reads documents, summarization reads community clusters — which is why Graph RAG's indexing cost (Q11) is dominated by these repeated LLM passes rather than the cheap embedding-only cost of a standard vector index. The community-detection step (Q3) is what gives Graph RAG its ability to answer broad "what are the themes here" questions that flat chunk retrieval structurally cannot, since no single chunk ever represents a cross-document theme the way a pre-built community summary does.

</details>

---

## Q14. What is the research origin of Graph RAG? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Graph RAG in the form this file describes was popularized by Microsoft Research (Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*, arXiv:2404.16130, 2024), which proposed the specific pipeline of LLM-based entity/relationship extraction, Leiden community detection, and LLM-generated community summaries at multiple hierarchy levels, released as the open-source `microsoft/graphrag` library.

The paper's core contribution is framing "global" query-focused summarization (questions about themes across an entire corpus, not a single passage) as a problem flat RAG cannot solve well by construction, and demonstrating that a pre-built community-summary hierarchy answers these queries substantially better than either flat vector retrieval or naive long-context stuffing. This same library later added LazyGraphRAG (#47) as a lower-cost mode, and LightRAG (#15) and HippoRAG (#20) are independent, differently-designed responses to the same original GraphRAG's high indexing cost.

</details>

---

## Q15. How does Graph RAG compare to LightRAG (#15)? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Both build an LLM-extracted knowledge graph rather than relying on flat chunk retrieval, but GraphRAG's indexing pipeline is substantially heavier: full entity/relationship extraction, Leiden community detection, and LLM summarization at every community, for the whole corpus, before any query arrives. LightRAG (#15) extracts entities and relationships too, but skips Leiden clustering and community summarization entirely, instead maintaining a simpler dual-level (local entity-anchored + global keyword-anchored) retrieval scheme directly over the extracted graph — a meaningfully cheaper indexing pipeline for a similar underlying goal.

The practical trade-off mirrors the general pattern across this bank's graph-based architectures: GraphRAG's heavier upfront investment produces genuinely pre-computed, inspectable global summaries (better for stable, high-repeat-query workloads, following the same logic as LazyGraphRAG's #47 comparison table); LightRAG's lighter approach trades some of that global-summary depth for lower cost and simpler incremental updates (#15 Q8), making it the more practical default when GraphRAG's full indexing cost isn't justified by query volume or repetition.

</details>

---

## Q16. What is the single distinctive mechanism that separates Graph RAG from standard vector RAG? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The distinctive mechanism is **indexing extracted entities and relationships as a structured graph, then pre-computing multi-level community summaries over that graph** — rather than indexing flat, independent chunks with no explicit representation of how entities across different chunks relate to each other. A vector index has no notion that "Company X" mentioned in document 3 and "Company X" mentioned in document 47 are the same entity with an accumulating set of facts; Graph RAG's entity resolution and graph construction step makes that connection explicit and queryable.

This single structural difference is what enables Graph RAG's headline capability (Q1, Q4): multi-hop and cross-document thematic questions that no single chunk answers become answerable by traversing the graph's explicit entity relationships or reading a pre-built community summary that already synthesized the cross-document pattern — a capability flat vector retrieval, which only ever compares one query embedding against independently-embedded chunks, cannot provide no matter how good the underlying embedding model is.

</details>

---

## Q17. What are the key tuning knobs for Graph RAG, and how do you choose them? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Knob | Effect | Starting point |
|---|---|---|
| Community detection resolution (Leiden algorithm parameter) | Higher resolution produces more, smaller communities (finer-grained but more numerous summaries); lower resolution produces fewer, larger communities | Tune against how naturally your corpus's entities cluster — a corpus with clearly separable topics needs less tuning than one with heavily interconnected entities |
| Community hierarchy depth | More levels give more granularity choices for the query router but multiply summarization cost | 2-3 levels is typical; deeper hierarchies mainly help very large, diverse corpora |
| Local vs. global routing threshold | Determines how a query is classified as needing entity-level (local) vs. theme-level (global) retrieval | Calibrate against a labeled query set the same way any routing decision in this bank is tuned (Adaptive RAG, #11) |
| Entity/relationship extraction prompt | Determines extraction quality and, indirectly, graph connectivity (Q19) | Iterate against a sample of manually-reviewed extractions before running at full corpus scale, since extraction errors compound into every downstream stage |

Community detection resolution is the knob most specific to Graph RAG among this list, and the hardest to get right without experimentation — it has no equivalent in flat vector RAG, and its right value is a property of your specific corpus's entity-relationship density rather than a value that transfers from another deployment's tuning.

</details>

---

## Q18. How do you decide when a query needs graph traversal vs. simple vector search in a hybrid Graph+Vector system? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The decision mirrors Adaptive RAG's (#11) query-complexity routing, applied specifically to the graph-vs-vector axis: queries asking about a specific, named fact ("what is Company X's revenue") are usually well-served by direct vector or entity lookup; queries asking about relationships, comparisons, or themes across entities ("how are Company X and Company Y connected," "what are the main risks discussed across this collection") need graph traversal or community-summary retrieval to answer well, since no single chunk contains the synthesized answer.

A practical router classifies incoming queries by these surface signals (single named entity + factual question type → vector/local; multiple entities, comparative/thematic language → graph/global) and falls back to running both paths and merging results when the classification is uncertain — the same fallback-to-both-paths pattern used for uncertain routing decisions elsewhere in this bank (e.g., LongRAG's #45 Self-Route). Track routing accuracy the same way as any other routing decision (Adaptive RAG's #11 Q18 evaluation methodology), since a Graph RAG deployment's cost profile depends heavily on not sending every query down the more expensive graph-traversal path by default.

</details>

---

## Q19. What is the characteristic failure mode of an over-fragmented or under-connected knowledge graph? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Over-fragmentation** (entity resolution failing to merge references to the same real-world entity, Q9) produces a graph where "Apple Inc.," "Apple," and "AAPL" exist as three separate, disconnected nodes, each accumulating only a fraction of the entity's actual relationships — multi-hop traversal starting from one variant misses relationships attached to the others, and community detection scatters what should be one entity's connections across multiple communities. **Under-connection** (relationship extraction missing genuine connections between entities that are related but never co-occur in a way the extraction prompt catches) leaves the graph sparser than the actual corpus's relationships would justify, causing multi-hop queries to dead-end at a node that should have had a path forward.

**Detection:** for over-fragmentation, audit high-degree or high-frequency entities for near-duplicate node names in the graph (a systematic scan, not manual case-by-case review, since fragmentation scales with corpus size); for under-connection, sample known ground-truth entity relationships from your source documents and check whether the graph actually contains an edge representing them. Both failure modes degrade Graph RAG's core value proposition silently — the system still returns *an* answer, just one built from an incomplete view of the entity's true connections, with no error surfaced anywhere in the pipeline to indicate the graph itself is the problem rather than the query or the generator.

</details>

---

## Q20. What are the limitations of Graph RAG, and how might the field evolve? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Current limitations: (1) **indexing cost is substantial and scales with corpus size** (Q11) — full entity/relationship extraction plus community summarization for every document is the most expensive indexing pipeline among this bank's graph-based architectures; (2) **entity resolution quality is a hard ceiling on graph usefulness** (Q9, Q19) — fragmentation or under-connection silently degrades multi-hop capability with no error signal; (3) **freshness is structurally harder than flat vector indexing** — a new document can require re-running community detection on affected communities, not just adding new isolated nodes; (4) **evaluation requires graph-specific metrics** (Q10) beyond standard retrieval metrics, which most teams have less tooling and experience for than standard RAG evaluation.

Likely evolution: this exact cost/limitation profile is precisely what motivated LazyGraphRAG's (#47) deferred-summarization approach and LightRAG's (#15) simpler dual-level retrieval — both are direct responses to Graph RAG's indexing cost, and a mature production deployment increasingly chooses among this family (GraphRAG, LightRAG, LazyGraphRAG, HippoRAG, #20) based on the specific cost/quality/freshness trade-off that matters most for its workload, rather than treating "Graph RAG" as a single fixed architecture — expect continued diversification within this family rather than convergence on one dominant graph-based RAG design.

</details>

---

## Q21. A genealogy startup needs to answer "how is my great-aunt related to this other family member?" over a modest family-tree graph. Does Graph RAG make sense at this scale? `[Basic]` `[Scenario]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The corpus here is small — thousands, not millions, of people and relationships — but the query type itself is inherently relational, which is what actually motivates Graph RAG (Q16), not corpus size. "How is X related to Y" has no single-chunk answer for flat vector search to return; it requires walking a relationship chain, which is exactly the capability a graph structure provides regardless of how big that graph is.

Because the source data is structured genealogical records (birth, marriage, and parentage records) rather than messy free text, build the entity/relationship graph directly from that structured data instead of running full LLM-based extraction (Q9) over documents — a more reliable and much cheaper path to a correct graph at this scale. Skip Leiden community detection and multi-level summarization (Q3) entirely; "what are the themes across this corpus" isn't the query pattern here, so that machinery would add cost without addressing the actual need. A direct graph-traversal or shortest-path query over the relationship graph answers the relationship-chain questions directly.

The trade-off: a small, structured-source graph is cheap to build and query today, but if the startup later ingests messier unstructured sources — old letters, scanned parish records — it will need the fuller entity-resolution and LLM-extraction machinery this file describes (Q9, Q19). Keep the query interface simple now; expect the ingestion pipeline, not the query pattern, to be what eventually needs to grow.

</details>

---

## Q22. A pharma company's drug-interaction assistant runs over a multi-million-node biomedical knowledge graph that gets a new ontology every night. What does production Graph RAG look like there? `[Advanced]` `[Scenario]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The hard constraints stack: graph scale in the millions of nodes, a nightly ontology/data refresh that has to land before the next business day, and a domain where a wrong drug-interaction answer is a patient-safety incident, not just a quality miss. Full Leiden community detection and re-summarization across the entire graph every night (Q11, Q17) is not affordable at this scale and cadence.

Use an incremental update strategy: identify which communities are actually touched by the night's ontology delta and re-cluster and re-summarize only those, rather than rebuilding the whole graph — the same cost-driven reasoning behind LightRAG's and LazyGraphRAG's cheaper designs (Q15, cross-referencing #15/#47), applied here as a scoped nightly job instead of a different architecture entirely. Resolve entities against canonical biomedical identifiers (RxNorm/UMLS-style codes) rather than free-text drug names, since Q19's over-fragmentation failure mode — the same compound ending up as several disconnected nodes — is a patient-safety issue in this domain, not merely an incompleteness issue. Every returned interaction claim should carry a provenance trail back to the specific graph edge and source study it came from.

What to monitor: whether the incremental nightly update actually completes before the next business day, an audit rate on high-degree drug nodes for entity-resolution drift, and clinician-review disagreement rate on flagged high-risk answers as an ongoing accuracy check. The trade-off: delta-scoped incremental re-clustering keeps nightly cost bounded, but it requires real engineering investment in dependency tracking (which communities a given ontology change actually affects) that a naive full-rebuild approach would never need to solve — justified here only because full nightly rebuilds are infeasible at this scale.

</details>

---

## Real-World Applications

| Application | Domain | Why Graph RAG Fits |
|---|---|---|
| Drug discovery & biomedical knowledge graph (e.g., BioGPT, Elsevier's Life Sciences KG) | Pharma / Biotech | Entities (genes, proteins, diseases, drugs) have dense relationships; graph traversal surfaces indirect connections that text search misses |
| Legal knowledge assistant | Legal | Case law is inherently graph-structured (citations, precedents, statutes); multi-hop reasoning finds how one ruling affects another |
| Fraud detection explainability | FinTech | Entity graph (accounts, transactions, people) enables "why was this flagged?" queries via relationship traversal |
| Enterprise organizational knowledge (org charts, project dependencies) | Enterprise | Who-knows-what and who-reports-to-whom queries require relationship traversal, not just document retrieval |
| Supply chain risk analysis | Manufacturing / Logistics | Supplier → component → product dependency graphs enable queries like "which products are affected if Supplier X fails?" |