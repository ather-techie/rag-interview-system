# 08 — Speculative RAG

> A smaller specialist model generates multiple draft answers; a larger generalist model selects and refines the best one — balancing quality and latency.

---

## Q1. What is Speculative RAG and what is the core insight behind it? `[Basic]`

**Answer:**

**Speculative RAG** (Wang et al., 2024) is inspired by speculative decoding in LLMs. The core insight is:

> A small, specialized model is faster and often better at *drafting* answers over retrieved documents. A large, general model is better at *judging and refining* those drafts.

**Pipeline:**

1. Retrieved documents are **partitioned into subsets**.
2. A **small specialist RAG drafter** generates one candidate answer per subset.
3. A **large generalist LLM** receives all candidate answers (not the raw documents) and selects/refines the best one.

This reduces the number of tokens the large LLM must process — instead of reading all retrieved chunks, it only reads short candidate answers — improving both **quality** and **latency**.

---

## Q2. How does document partitioning work in Speculative RAG? `[Intermediate]`

**Answer:**

Rather than passing all retrieved chunks to a single model, Speculative RAG partitions them:

1. Retrieve top-k chunks (e.g., k=10).
2. Partition into m subsets of size k/m (e.g., 5 subsets of 2 chunks each).
3. Each subset is processed independently by the drafter model to produce a candidate answer with a rationale.

**Why partition?**
- Reduces context length per drafter call (faster inference).
- Different subsets may contain complementary evidence — multiple drafts capture this diversity.
- Prevents the "lost-in-the-middle" problem since each drafter sees fewer chunks.

The partitioning can be random, diversity-maximized (pick chunks that cover different subtopics), or evidence-ranked.

---

## Q3. What is the role of the large generalist LLM in Speculative RAG? `[Intermediate]`

**Answer:**

The large LLM acts as a **verifier and refiner**, not a reader of raw documents. It receives:

```
Candidate 1: [Answer A with rationale from subset 1]
Candidate 2: [Answer B with rationale from subset 2]
...
Candidate m: [Answer N with rationale from subset m]

Task: Select the most accurate, well-supported answer, or synthesize a better answer combining the candidates.
```

**Its responsibilities:**
1. **Judge faithfulness** — Which candidate is best grounded in evidence?
2. **Synthesize** — Combine complementary evidence from multiple candidates.
3. **Refine** — Improve fluency, completeness, or reasoning of the best candidate.

Because it only reads candidate answers (not raw chunks), the large model's context is much shorter — typically 3–5x fewer tokens than standard RAG.

---

## Q4. How does Speculative RAG compare to Speculative Decoding in standard LLM inference? `[Advanced]`

**Answer:**

Both share the **draft-then-verify** philosophy, but apply it differently:

| Aspect | Speculative Decoding | Speculative RAG |
|---|---|---|
| **What's drafted** | Token sequences | Full answers |
| **Drafter** | Small version of same model | Small specialist RAG model |
| **Verifier** | Large version of same model | Large generalist LLM |
| **Goal** | Reduce decoding latency | Reduce context length + improve accuracy |
| **Acceptance criterion** | Token probability alignment | Answer quality / faithfulness score |
| **Granularity** | Token-level | Answer-level |

In Speculative Decoding, the large model *accepts or rejects* the small model's tokens. In Speculative RAG, the large model *selects among* the small model's answers. The analogy is intentional — both exploit the asymmetry between fast drafting and accurate verification.

---

## Q5. When would Speculative RAG outperform Agentic RAG, and when would it not? `[Advanced]`

**Answer:**

**Speculative RAG wins when:**
- Queries can be answered from a single retrieval round (no multi-hop needed).
- You have a large retrieved context and want to reduce large-LLM token costs.
- Latency matters — parallel drafting + single verification is faster than sequential agent loops.
- You have a domain-specific drafter model fine-tuned on your corpus.

**Agentic RAG wins when:**
- The query requires **multi-hop reasoning** (follow-up retrievals based on intermediate findings).
- The query type is **unknown at retrieval time** (agent decides which tool to call).
- The knowledge base doesn't contain all needed information (agent can fall back to web search).
- You need **interactive workflows** where the LLM takes real-world actions (not just Q&A).

**In practice:** Speculative RAG is best for a well-scoped, document-heavy Q&A product. Agentic RAG is better for open-ended assistants and workflows. Many production systems combine both — use Speculative RAG for the retrieval-heavy path and Agentic RAG for complex, multi-step queries.
