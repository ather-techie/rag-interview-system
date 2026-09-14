# 34 — Privacy-Preserving RAG

> How to build RAG systems that retrieve relevant information without exposing the corpus, user queries, or embeddings to untrusted parties.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
INGESTION (offline, per silo)
  Raw Documents
        │
        ▼
  Anonymizer (NER + regex PII scrub)
        │
        ▼
  Embedder ──► Local Vector Index

QUERY (online)
  Client Query
        │
        ▼
  On-device Embedder (local model, raw text never leaves the client)
        │
        ▼
  DP Noise Injector (calibrated Gaussian/Laplace noise, budget ε)
        │
        ▼
  k-Anonymity Query Obfuscator (optional: real query + k−1 dummy queries)
        │
        ▼
  Federated Retrieval Coordinator
        │
        ├─► Silo A index ─┐
        ├─► Silo B index ─┤── merge via Reciprocal Rank Fusion (RRF)
        └─► Silo C index ─┘
        │
        ▼
  Ranked Results (server never saw raw query text or the full corpus)
```

### Key Components

| Component | Responsibility |
|---|---|
| On-device Embedder | Converts the query to a vector locally so raw query text never reaches the server |
| DP Noise Injector | Adds calibrated noise to the query embedding to defeat vec2text-style inversion attacks |
| Query Obfuscator | Wraps the real query with k−1 dummy queries so the server cannot tell which result was wanted |
| Federated Retrieval Coordinator | Dispatches the query to per-silo indexes in parallel and merges ranked lists via RRF without seeing raw content |
| Anonymizer (ingestion-time) | Scrubs PII from documents via NER/regex before they are embedded and indexed |
| Trust / Access Layer | Enforces per-tenant and per-silo authorization on top of the privacy techniques above |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| On-device embedding | sentence-transformers, ONNX Runtime (mobile/edge inference) |
| PII anonymization | spaCy NER, regex pattern libraries, Microsoft Presidio |
| Differential privacy | Custom Laplace/Gaussian noise implementation, Opacus (DP-trained models) |
| Federated retrieval | Custom async coordinator (asyncio), gRPC between silo services |

---

## Q1. What is Privacy-Preserving RAG and what problem does it solve? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Privacy-preserving RAG addresses a gap that multi-tenancy and ACL controls don't cover: in standard RAG, the retrieval server sees every query and every retrieved document in the clear. In regulated industries (healthcare, finance, legal), that creates concrete risk even when access control is correctly implemented:

- A vendor-hosted vector DB sees raw patient queries ("what is my HIV test result?").
- Embeddings can be approximately inverted back to the original text.
- Query logs reveal user intent even when document access is properly scoped.

Privacy-preserving RAG combines several techniques — on-device embedding, differential privacy noise, query obfuscation, federated retrieval, and pre-indexing anonymization — to reduce what any single party (including the retrieval provider itself) can learn about the user or the corpus. It is the RAG answer to "what if the server operator is not fully trusted?"

</details>

---

## Q2. What is the single mechanism that separates Privacy-Preserving RAG from RAG with just encryption and access control? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Standard security controls (TLS, encryption at rest, ACLs) assume the retrieval server is trusted to see plaintext queries and documents, and only restrict *who else* can see them. Privacy-preserving RAG instead assumes **the server itself must not learn** certain things:

| Layer | What it hides from the server |
|---|---|
| On-device embedding | Raw query text (server only ever sees a vector) |
| Differential privacy noise | The exact embedding (defeats inversion attacks even on the vector) |
| k-anonymity obfuscation | Which of k queries sent was the real one |
| Federated retrieval | The full corpus (each silo only sees its own documents) |
| Pre-indexing anonymization | PII inside documents, even from someone with raw index access |

Encryption and ACLs are still necessary — they stop unauthorized third parties. Privacy-preserving RAG is the additional layer for when the *authorized* operator itself must be prevented from learning too much. Production systems typically need both.

</details>

---

## Q3. Walk through the privacy threat model for a RAG pipeline — what can each party learn, and what mitigates it? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Threat | Example | Mitigation |
|--------|---------|-----------|
| **Query exposure** | Vendor sees raw user query | Query obfuscation, on-device embedding |
| **Corpus exposure** | Retrieval API reveals document content | Federated retrieval, blind retrieval |
| **Embedding inversion** | Embeddings reconstructed to approximate text | Differential privacy on embeddings |
| **Membership inference** | Attacker infers whether a document is in the corpus | DP training for embedding models |
| **Cross-tenant leakage** | Tenant A's query retrieves Tenant B's data | Per-tenant index isolation (see multi-tenancy guide) |

The pattern across every row: assume the party that would normally be trusted (the retrieval server, the embedding model provider, the shared vector index) is instead a threat actor, and ask what it could still learn from what it necessarily sees. Query exposure and embedding inversion are about the *query* path; corpus exposure and membership inference are about the *indexed data*; cross-tenant leakage is about *isolation* between callers of the same system. Each mitigation in the right column maps to exactly one of the five techniques covered below.

</details>

---

## Q4. How does on-device embedding move query privacy from the server to the client? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Instead of sending the raw query string to the retrieval server, the query is embedded locally on the client (browser, mobile app, edge device) using a small model, and only the resulting vector is transmitted:

```
Without on-device embedding:
  Client ──► "What is my HIV test result?"  ──► Retrieval Server (sees query)

With on-device embedding:
  Client ──► [0.21, -0.14, 0.88, ...]  ──────► Retrieval Server (sees only vector)
```

```python
# Client-side (on device, never leaves the device)
from sentence_transformers import SentenceTransformer

LOCAL_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")  # small enough for on-device

def embed_locally(query: str) -> list[float]:
    emb = LOCAL_MODEL.encode(query, normalize_embeddings=True)
    return emb.tolist()

# Only the vector is sent to the server
query_vector = embed_locally("What is my HIV test result?")
results = retrieval_server.search(query_vector, k=5)  # server never sees raw text
```

This is the cheapest privacy win available: no infrastructure change on the server, just a model swap on the client, and it fully removes plaintext queries from any server log. Its limit is that the vector itself is not private — with vec2text-style inversion attacks, an adversary who knows the embedding model can approximately reconstruct the original text from the vector alone. That's why on-device embedding is almost always paired with differential privacy noise (Q8) rather than deployed alone.

</details>

---

## Q5. How does federated retrieval work across data silos, and how are ranked lists merged without a central party seeing raw content? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

In federated retrieval, each data silo (a hospital, a bank branch, a regional office) keeps its own local retrieval index. A central coordinator fans a query out to every silo in parallel and merges the ranked lists it gets back — it never receives raw documents, only `(doc_id, score)` pairs.

```
Traditional (centralized):
  All Documents ──► Central Index ──► All Queries routed here

Federated:
  Hospital A index ──┐
  Hospital B index ──┤──► Coordinator (merges ranked lists via RRF)
  Hospital C index ──┘

  Each index sees only its own queries (routed by the coordinator)
  Coordinator sees only ranked doc IDs + scores — no raw content
```

```python
import asyncio
from typing import NamedTuple

class RankedResult(NamedTuple):
    silo_id: str
    doc_id: str
    score: float

async def federated_retrieve(query_vector, silos, k: int = 5) -> list[RankedResult]:
    """Query each silo in parallel; merge results with Reciprocal Rank Fusion."""

    async def query_silo(silo):
        results = await asyncio.to_thread(silo.search, query_vector, k * 2)
        return [(silo.id, r["doc_id"], r["score"]) for r in results]

    all_results = await asyncio.gather(*[query_silo(s) for s in silos])

    rrf_scores = {}
    for silo_results in all_results:
        for rank, (silo_id, doc_id, _) in enumerate(silo_results):
            key = (silo_id, doc_id)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + rank + 1)

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [RankedResult(silo_id=k[0], doc_id=k[1], score=v) for k, v in merged[:k]]
```

RRF (`1 / (60 + rank)`, summed across silos) is used instead of raw score averaging because similarity scores are not comparable across independently-trained or independently-scaled silo indexes; rank position is. The coordinator's trust boundary is the whole point: it can be run by a third party (or none of the silos) and still never sees a document.

</details>

---

## Q6. How does Privacy-Preserving RAG differ from multi-tenant ACL-based RAG? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Multi-tenant ACL controls *who can access what* — the server is trusted, and it correctly enforces access rules between tenants. Privacy-preserving RAG addresses the case where *the server itself is untrusted* (or must not see certain data), regardless of whether it enforces tenant boundaries correctly.

Example: a SaaS retrieval vendor should not see raw patient queries even if it flawlessly enforces per-hospital tenant isolation. ACL plus encryption handles cross-tenant leakage (Tenant A cannot see Tenant B's data). On-device embedding plus differential privacy handles what the vendor itself can infer about any tenant's queries.

In practice, production systems need both layers: ACL for authorization between callers, privacy techniques for data minimization against the operator. Neither substitutes for the other — a system with perfect ACLs and no privacy techniques still leaks every query to the vendor; a system with perfect on-device privacy and no ACLs still leaks Tenant A's documents to Tenant B.

</details>

---

## Q7. When is on-device embedding plus DP noise sufficient, and when do you actually need federated retrieval? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Situation | Sufficient technique |
|---|---|
| One organization, one shared corpus, vendor-hosted retrieval | On-device embedding + DP noise — the corpus can live in one index; only the *query* needs hiding |
| Data legally cannot leave its originating org (hospital-to-hospital, bank-to-bank, cross-border) | Federated retrieval — the *corpus itself* cannot be centralized, regardless of query privacy |
| Regulator requires provable per-silo data residency | Federated retrieval, silo-local encryption at rest |
| Internal team, trusted retrieval infra, external attacker is the concern | On-device embedding + DP noise is enough; federation adds cost with no benefit |

The deciding question is *where the constraint lives*. If the constraint is "the query must not be readable by the server," on-device embedding and DP noise solve it with low complexity and no architecture change to the corpus. If the constraint is "the documents themselves must never leave organization X," no amount of query-side privacy helps — you need federated retrieval, which is the higher-complexity, higher-latency option (Q15) and should be reserved for when data residency, not query confidentiality, is the actual requirement.

</details>

---

## Q8. How do you implement differential-privacy noise injection for query embeddings? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The embedding of a query is a high-dimensional vector. An adversary with the vector and knowledge of the embedding model can use vec2text-style inversion attacks to approximately reconstruct the original text. Differential privacy (DP) adds calibrated Gaussian noise to the embedding before it leaves the client, so the noisy vector is (ε, δ)-indistinguishable from the embedding of any neighboring query (one differing by a word). The server's retrieval result shifts slightly — because the query vector moved — but the client's privacy has a formal guarantee.

```python
import numpy as np

def privatize_embedding(emb: np.ndarray, epsilon: float = 1.0, sensitivity: float = 1.0) -> np.ndarray:
    """Add Gaussian noise calibrated to (epsilon, delta=1e-5)-DP."""
    delta = 1e-5
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = np.random.normal(0, sigma, size=emb.shape)
    noisy = emb + noise
    return noisy / np.linalg.norm(noisy)  # re-normalize so cosine similarity still works


def private_retrieval(query: str, epsilon: float = 1.0, k: int = 5):
    emb = embed_locally(query)
    noisy_emb = privatize_embedding(np.array(emb), epsilon=epsilon)
    return retrieval_server.search(noisy_emb.tolist(), k=k)
```

The trade-off is controlled entirely by `epsilon`: smaller epsilon means more noise, stronger privacy, and lower recall (see Q10 for concrete numbers). `epsilon=1.0` is a common production starting point, giving roughly a 5% recall drop for a formal indistinguishability guarantee.

</details>

---

## Q9. How do you implement k-anonymity query obfuscation, and what does it cost? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Instead of (or in addition to) noising the real query, send the real query embedding alongside `k−1` dummy queries at a random position, so the server cannot tell which of the `k` results the client actually wanted:

```python
import numpy as np

def obfuscate_query(real_emb: np.ndarray, k_anon: int = 4, noise_scale: float = 0.1):
    """Return k query vectors; the real query sits at a random index."""
    dummies = [real_emb + np.random.normal(0, noise_scale, real_emb.shape) for _ in range(k_anon - 1)]
    dummies = [d / np.linalg.norm(d) for d in dummies]

    real_index = np.random.randint(0, k_anon)
    queries = dummies[:real_index] + [real_emb] + dummies[real_index:]
    return queries, real_index


def k_anonymous_retrieve(query: str, k_anon: int = 4) -> list:
    real_emb = np.array(embed_locally(query))
    queries, real_idx = obfuscate_query(real_emb, k_anon=k_anon)
    all_results = [retrieval_server.search(q.tolist(), k=5) for q in queries]
    return all_results[real_idx]  # client discards the k-1 dummy result sets locally
```

The cost is direct: the server does `k_anon`× the retrieval work for every real query, and the client does `k_anon`× the network round trips. Unlike DP noise, k-anonymity obfuscation causes **zero retrieval-quality loss** for the real query (its embedding is sent unmodified) — the entire cost is server load, not accuracy. It composes with DP noise: you can noise the real query *and* wrap it in k dummies for layered protection.

</details>

---

## Q10. What are the key tuning knobs for Privacy-Preserving RAG, and how do you choose them? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Technique | Privacy Guarantee | Retrieval Impact | Complexity |
|-----------|------------------|-----------------|-----------|
| On-device embedding | Query text hidden from server | None | Low |
| DP noise (ε) | Embedding ≈ uninvertible | Recall impact scales with 1/ε | Low |
| Federated retrieval | Each silo sees only its own docs | Moderate (RRF merge) | High |
| Anonymization before indexing | PII not stored in index | Minor (entity loss) | Medium |
| k-Anonymity queries (k) | Query identity hidden among k | None (k× server load) | Low |

The three knobs to actually tune in production:

```
epsilon = 0.1 → strong privacy, ~15% recall drop
epsilon = 1.0 → moderate privacy, ~5% recall drop   ← common production choice
epsilon = 10  → weak privacy, ~0.5% recall drop
```

- **`epsilon`** — start at 1.0 and move down only as far as your recall SLA allows; below ~0.3 the recall drop is usually unacceptable for anything but the most sensitive query classes.
- **`k_anon`** — 4–8 is typical; higher k gives stronger anonymity-set guarantees at linear server-cost growth, with diminishing privacy return above ~10.
- **`noise_scale`** for dummy queries — must be large enough that dummies are not trivially distinguishable from the real query by embedding-space distance, but small enough that all k results returned look plausible (an obviously nonsensical dummy query defeats the anonymity set).

Tune these against a held-out recall benchmark (Q11), not in isolation — the three knobs interact (heavier DP noise plus a small k_anon can make the real query's dummies statistically distinguishable again).

</details>

---

## Q11. How do you evaluate whether your DP noise and obfuscation settings preserve acceptable retrieval quality? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Build a fixed evaluation set of representative queries with known-relevant documents (the same golden set used for standard retrieval evaluation), then compare recall@k and NDCG with privacy techniques on vs. off:

1. Run every query through the plain (non-private) pipeline; record recall@5, recall@10, NDCG@10 as baseline.
2. Re-run the same queries through the privatized pipeline (on-device embedding → DP noise at your candidate ε → obfuscation at your candidate k) against the same index.
3. Compute the delta in recall/NDCG. Plot delta vs. ε across a sweep (0.1, 0.5, 1.0, 2.0, 10) to find your organization's actual privacy-utility curve rather than relying on published numbers, which depend on embedding model and corpus geometry.
4. Segment by query sensitivity class if you have one — e.g., queries flagged as containing health terms might warrant a lower ε (stronger privacy, larger accepted recall loss) than general product-FAQ queries.

Treat the resulting curve as a per-deployment artifact, not a one-time calculation: re-run it whenever the embedding model, corpus, or index size changes materially, since recall sensitivity to noise depends on how tightly clustered the corpus's embeddings are.

</details>

---

## Q12. How would you build a decision-gate evaluation to certify a Privacy-Preserving RAG deployment meets a target epsilon while holding an SLA recall? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A decision gate turns the recall-vs-epsilon curve from Q11 into a pass/fail check that can block a deployment, not just inform one:

```
1. Define two thresholds up front, signed off by both security and product:
   - epsilon_max  (the loosest privacy budget legal/compliance will accept)
   - recall_min   (the lowest recall@10 product will accept, e.g. 90% of non-private baseline)

2. Nightly (or per-PR) CI job:
   for epsilon in sweep(epsilon_max down to 0.1):
       run golden eval set through privatized pipeline at this epsilon
       record recall@10
   find epsilon* = the *largest* epsilon <= epsilon_max that still meets recall_min
       (largest epsilon = weakest acceptable noise, i.e., best utility inside the privacy floor)

3. Gate: if no epsilon in [0.1, epsilon_max] meets recall_min -> FAIL the build.
   This means the current embedding model / corpus cannot satisfy both constraints
   simultaneously, and requires action (better embedding model, corpus dedup to
   tighten clusters, or renegotiating recall_min with product) rather than silently
   shipping a system that is either too leaky or too inaccurate.

4. Record epsilon* in the deployment manifest; the query-time DP noise injector
   reads it from config, so security can audit "what epsilon is live in prod"
   without reading application code.
```

The key discipline is refusing to treat epsilon as a single global constant chosen once — corpus and model changes shift the curve, and a gate that re-derives `epsilon*` on every relevant change catches silent privacy or utility regressions before they reach production, the same way a golden-set regression gate catches retrieval-quality regressions in a non-private system.

</details>

---

## Q13. What is the characteristic failure mode of DP-noised retrieval, and how do you detect it in production? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The characteristic failure is **over-noising**: an epsilon chosen too aggressively (too small) for the corpus's embedding geometry, so the noisy query vector drifts outside the neighborhood of its true nearest documents entirely. Symptoms in production:

- User-facing: "I searched for X and got completely unrelated results" complaints spike after a privacy-budget change.
- Metrics: recall@k drops sharply for a subset of query types — typically short, generic queries are hit hardest, because their embeddings sit in denser regions of the space where a fixed noise magnitude covers proportionally more "wrong neighbor" territory than for long, specific queries.
- A/B signal: click-through / thumbs-up rate on retrieved results falls for the private-pipeline cohort relative to a non-private control, even though nothing else about the corpus changed.

Detection: track recall@k (against a small labeled sample, or via implicit signals like re-query rate within the same session) segmented by epsilon value if you support per-tenant or per-sensitivity-class epsilon. A sudden regression correlated with an epsilon change (not a corpus or model change) is the signature. The fix is almost always to move epsilon up incrementally and re-run the Q11 evaluation rather than disabling DP noise outright.

</details>

---

## Q14. What is the characteristic failure mode of pre-indexing anonymization, and how do you debug it? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Anonymization runs NER plus regex over documents before they are embedded and indexed:

```python
import re
import spacy

nlp = spacy.load("en_core_web_sm")

PII_PATTERNS = {
    "SSN":   r"\b\d{3}-\d{2}-\d{4}\b",
    "Email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "Phone": r"\b(\+1)?\s*\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
}

def anonymize_text(text: str) -> str:
    for label, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[{label}]", text)
    doc = nlp(text)
    for ent in reversed(doc.ents):  # reversed to preserve character offsets
        if ent.label_ in {"PERSON", "ORG", "GPE", "LOC"}:
            text = text[:ent.start_char] + f"[{ent.label_}]" + text[ent.end_char:]
    return text
```

Its characteristic failure is **silent under-redaction**: regex patterns miss format variants (an SSN written `123 45 6789` instead of `123-45-6789`; an international phone number; an email with a `+` alias), and NER misses PII that doesn't look like a named entity to the model — a patient ID embedded in a sentence, a rare name the NER model wasn't trained on, PII split across a sentence boundary by chunking. Because the scrubbed text is what gets embedded and indexed, a miss here means PII is now permanently baked into the vector index, not just present in a log that could be purged.

Debugging playbook: (1) run a held-out set of documents with known, deliberately-varied PII formats through the anonymizer and measure recall of redaction, not just precision; (2) add a post-anonymization regex "canary" sweep across a broader PII pattern library (e.g., Microsoft Presidio's recognizer set) as a second pass, purely for detection/alerting, even if the first pass used a narrower custom set; (3) treat any anonymization miss found in production as requiring a full re-index of the affected documents, since the embedding itself may already leak information about the original text (Q8's inversion risk applies to *any* stored embedding, not just query embeddings).

</details>

---

## Q15. What is the cost and latency overhead of k-anonymity obfuscation and federated retrieval at scale? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**k-anonymity obfuscation:** cost scales linearly with `k_anon`. At `k_anon=4`, every real query becomes 4 retrieval calls; at 10M queries/month and $0.0003/query illustrative retrieval cost, baseline is $3,000/month and obfuscated cost is $12,000/month — a flat `(k_anon - 1) × baseline` tax, with no accuracy trade-off to offset it. This is the right technique only when the marginal cost is acceptable relative to the value of hiding query identity.

**Federated retrieval:** cost is dominated by fan-out latency, not request volume. A query that previously hit one index now hits N silos in parallel — wall-clock latency is bounded by the *slowest* silo, not the average, so p99 latency degrades faster than p50 as silo count grows or as any single silo's index gets slow or overloaded. Illustrative: 5 silos each with 50ms p50 / 300ms p99 individually; the federated query's p99 approaches the max of 5 independent p99 draws, commonly pushing federated p99 latency 1.5–2x higher than any single silo's own p99, even though average latency barely moves.

Mitigations: for k-anonymity, cap `k_anon` to the minimum that meets your anonymity-set requirement rather than over-provisioning; for federated retrieval, set a per-silo timeout shorter than your overall SLA and merge whatever silos responded in time (with a fallback for degraded results) rather than blocking on the slowest one — accepting a small recall loss from a timed-out silo is usually preferable to failing the whole query.

</details>

---

## Q16. How do you implement the right-to-erasure (GDPR Art. 17) requirement in a Privacy-Preserving RAG system? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Every vector stored in any index — including per-silo indexes in a federated deployment — must be traceable to its source document via a `source_doc_id` metadata field set at ingestion time. To erase a document:

1. Identify which silo(s) hold vectors derived from the document (in a federated system, the coordinator does not know this by design, so the erasure request must be routed to each silo, or the silo mapping must be tracked separately from the retrieval path).
2. Query each affected silo's vector DB for all vectors with `source_doc_id = X`; delete those vectors by ID.
3. Delete the raw document and its parsed chunks from the document store.
4. Delete any per-user semantic memory entries that reference that document.

The hard part is step 4: if a user's conversational memory contains a *derived* claim ("the report mentioned X") rather than the document itself, that derived entry is also subject to erasure, but it no longer has an obvious link back to `source_doc_id` once it has been paraphrased into memory. Maintain an explicit mapping from `source_doc_id` to every derived memory ID at write time — never try to reconstruct this relationship after the fact by re-scanning memory content, which is unreliable and auditable gaps remain (a regulator asking "prove document X is fully erased" needs a positive list, not a best-effort search). In a federated architecture, this erasure protocol must run per-silo since no single party has global visibility into where a document's vectors ended up.

</details>

---

## Q17. What attack surfaces remain even after applying all five privacy techniques, and how do you mitigate them? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Layering on-device embedding, DP noise, k-anonymity obfuscation, federated retrieval, and pre-indexing anonymization closes the obvious channels, but several residual surfaces remain:

- **Traffic/timing analysis** — even with obfuscated queries, an adversary observing network traffic can potentially correlate query timing and size with user activity (e.g., a burst of queries right after a known event). Mitigate with constant-size padding and randomized send delays for dummy queries.
- **Colluding silos** — federated retrieval assumes silos don't share what they see; if two silos collude, they can compare which queries they each received and partially reconstruct cross-silo access patterns. Mitigate with silo-level query anonymization (route via a mix network or add per-silo dummy traffic) rather than relying on organizational trust alone.
- **Membership inference on the embedding model** — even a DP-noised query embedding can leak whether a *specific* document was in the training or fine-tuning set of the embedding model itself, independent of the retrieval index. Mitigate by using a DP-trained embedding model (e.g., via Opacus) when the embedding model itself was fine-tuned on sensitive data.
- **Result-side leakage** — the *response* (retrieved chunks, or the LLM's generated answer) can itself reveal information about the corpus even when the query path is fully private. This is not addressed by any of the five techniques above and requires separate output-side controls (response filtering, differential privacy on aggregate statistics).

The practical takeaway: privacy-preserving RAG techniques are about the query and index path specifically; they compose with, but do not replace, output-side controls and organizational/legal safeguards against collusion.

</details>

---

## Q18. Design a privacy-preserving RAG system for a multi-hospital healthcare consortium. `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Requirements:** each hospital's patient records must never leave its own infrastructure (data residency); clinician queries must not reveal patient identity to a shared coordinator; the system must support GDPR/HIPAA-style erasure per patient.

```
Per-hospital (silo), on-prem:
  Patient Records → Anonymizer (strip direct identifiers, keep source_doc_id)
                   → Embedder → Local Vector Index (encrypted at rest)

Clinician query path:
  Clinician device → On-device embed → DP noise (epsilon tuned per query
                    sensitivity class) → k-anonymity wrap (k=4-8)
                    → Federated Coordinator (cross-org, low-trust)
                        ├─► Hospital A silo (own infra)
                        ├─► Hospital B silo (own infra)
                        └─► Hospital C silo (own infra)
                    → RRF merge → ranked doc IDs + scores only
                    → Clinician device resolves doc IDs against its own
                      hospital's document store for final display
```

Key design decisions: (1) the coordinator is run by a neutral third party or a consortium-governed service, never by any single hospital, since it is the one component that sees cross-org query patterns; (2) each hospital retains full custody of its own vector index and raw documents — nothing crosses the network except query vectors and ranked `(silo, doc_id, score)` tuples; (3) erasure requests are routed per-hospital using the `source_doc_id` scheme (Q16), since no central registry of all patients across hospitals should exist; (4) per-query epsilon is set higher (more privacy, lower recall) for queries flagged with sensitive-topic terms, and lower for general operational queries, using the tuning framework from Q10 and Q11's evaluation harness to keep both compliance and clinician usability defensible.

</details>

---

## Q19. What is the research and practical origin of these privacy techniques? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Privacy-preserving RAG borrows from three separate research lineages rather than being a single paper's invention:

- **Differential privacy** — Dwork et al.'s foundational formalization (see Dwork & Roth, *The Algorithmic Foundations of Differential Privacy*, 2014) defines the (ε, δ)-indistinguishability guarantee used to calibrate the Gaussian noise in Q8. This is the same mathematical framework used for DP-SGD in private model training, applied here to query embeddings instead of gradients.
- **Federated learning / federated computation** — McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data* (arXiv:1602.05629, 2017), established the pattern of keeping data local and only exchanging model updates or, in the retrieval case, ranked results. Federated retrieval is a direct application of this "compute locally, aggregate centrally" principle to search rather than training.
- **Embedding inversion attacks** — Morris et al., *Text Embeddings Reveal (Almost) As Much As Text* (arXiv:2310.06816, 2023), demonstrated that dense embeddings can be inverted back to near-verbatim text (the "vec2text" line of work), which is the specific threat that motivates adding DP noise to embeddings rather than treating the vector as safe by default.

Privacy-preserving RAG is best understood as an applied synthesis of these three lines, assembled into a retrieval pipeline, rather than a single named architecture with one canonical paper.

</details>

---

## Q20. What are the limitations of current privacy-preserving RAG techniques, and how is the field likely to evolve? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Current limitations:

- **Utility cost is real, not theoretical** — every technique here trades measurable recall or latency for privacy; there is no free lunch, and teams that treat epsilon as a checkbox rather than a tuned parameter (Q10–Q12) end up either leaking more than intended or shipping a noticeably worse product.
- **No protection for the generation step** — all five techniques protect the *retrieval* path; the LLM's generated answer can still leak corpus information if it verbatim-quotes a retrieved chunk, and none of the retrieval-side privacy budget accounts for this.
- **Federated retrieval's coordinator remains a partial trust bottleneck** — even seeing only ranked doc IDs, a sufficiently persistent coordinator can build a query-pattern profile over time (Q17).
- **DP guarantees are per-query, not cumulative** — repeated queries from the same user against the same corpus compose (each query "spends" privacy budget), and most production deployments do not track cumulative epsilon spend per user the way rigorous DP systems require.

Likely evolution: **secure multi-party computation (MPC)** and **homomorphic encryption** for retrieval are the research frontier for removing the federated coordinator's partial-trust role entirely — enabling similarity search over encrypted vectors without any party seeing plaintext embeddings, at the cost of orders-of-magnitude higher compute per query today. Expect these to remain impractical for high-QPS production use for several more years, with DP noise plus federated retrieval remaining the pragmatic default, and generation-side leakage controls (citation-scoped answers, output filtering) becoming a more explicit second half of the privacy story as retrieval-side techniques mature.

</details>

---

## Compliance Considerations

| Regulation | Requirement | RAG Implication |
|-----------|-------------|----------------|
| **HIPAA** | PHI must be protected at rest and in transit | Anonymize before indexing; encrypt vectors at rest |
| **GDPR Art. 17** | Right to erasure | Must be able to delete all vectors derived from a document (Q16) |
| **GDPR Art. 25** | Privacy by design | On-device embedding as default; no raw query logging |
| **CCPA** | User data opt-out | Per-user semantic memory must be deletable on request |

## Real-World Applications

| Application | Domain | Why Privacy-Preserving RAG Fits |
|---|---|---|
| Cross-hospital clinical decision support | Healthcare | Patient records cannot be centralized across institutions; federated retrieval keeps data in place while still enabling shared search |
| Vendor-hosted enterprise search | Enterprise SaaS | The retrieval vendor must not be able to read customer queries or documents, even though it operates the infrastructure |
| Financial fraud investigation across banks | Finance | Banks cannot share raw transaction records but need to jointly search for patterns; federated retrieval plus DP noise satisfies both constraints |
| Legal e-discovery platforms | Legal | Privileged documents must be searchable without exposing content to the platform operator |
| Consumer health / wellness assistants | Consumer | On-device embedding keeps sensitive personal queries off any server by default, satisfying "privacy by design" expectations |
