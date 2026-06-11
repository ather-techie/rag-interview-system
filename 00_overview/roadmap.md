# Roadmap: RAG Maturity Progression

> From your first RAG prototype to a production-grade retrieval system — a practitioner's progression map.

---

## The RAG Maturity Model

Every production RAG system starts simple and grows in complexity. This five-level model maps that progression and identifies the forcing functions that push teams from one level to the next.

```
Level 0: Proof of Concept
     │
     ├─► reaches: ~100 documents, latency <1s, accuracy <60%
     ├─► team size: 1–2
     └─► next level forced by: "It works on my test set but fails on real queries"
                ▼
Level 1: Advanced Naive RAG
     │
     ├─► reaches: ~10K documents, latency <500ms, accuracy 65–75%
     ├─► team size: 2–3
     └─► next level forced by: "We need to handle domain-specific retrieval and cost control"
                ▼
Level 2: Modular / Adaptive RAG
     │
     ├─► reaches: ~100K documents, latency ~300ms, accuracy 75–85%
     ├─► team size: 3–5
     └─► next level forced by: "Some queries need reasoning; static retrieval is not enough"
                ▼
Level 3: Agentic / Specialized RAG
     │
     ├─► reaches: ~1M documents, latency 200–2000ms (variable), accuracy 80–90%
     ├─► team size: 5–10
     └─► next level forced by: "We need systems that improve themselves based on user feedback"
                ▼
Level 4: Self-Improving RAG
     │
     ├─► reaches: unbounded scale, adaptive accuracy, multi-domain
     ├─► team size: 10+
     └─► requires: fine-tuning, feedback loops, evaluation infrastructure
```

| Level | Typical Team Size | Corpus Size Threshold | Latency Budget | Primary Failure Mode |
|-------|-------------------|-----------------------|-----------------|----------------------|
| 0 | 1–2 | <1K docs | <1s | Poor recall on real queries |
| 1 | 2–3 | 10K docs | <500ms | Domain-specific terminology; cost explosion |
| 2 | 3–5 | 100K docs | <300ms | Retrieval alone insufficient for complex queries |
| 3 | 5–10 | 1M docs | 200–2000ms | Inconsistent quality; no feedback loop |
| 4 | 10+ | Unbounded | Variable | Requires sustained ML ops; technical debt |

---

## Skill Prerequisites by Level

Before advancing to the next maturity level, ensure you have these skills.

| Level | Required Skills | Secondary Skills | Knowledge Gap Resources |
|-------|-----------------|------------------|--------------------------|
| 0 | Python async/await, vector math (dot product, cosine), basic prompt engineering | SQL queries, JSON parsing | `01_concepts/embeddings.md` |
| 1 | Vector DB operations, chunking strategies, retrieval evaluation metrics | Information retrieval fundamentals, LLM fine-tuning basics | `01_concepts/chunking_strategies.md`, `01_concepts/evaluation_metrics.md` |
| 2 | Query understanding, reranking, system design (caching, rate limiting), observability | Graph theory, entity linking | `01_concepts/reranking.md`, `00_overview/system_design_principles.md` |
| 3 | Agent framework design, multi-step reasoning, failure recovery, state management | Reinforcement learning, reward modeling | `02_interview_bank/04-agentic-rag.md` |
| 4 | ML-based evaluation (RAGAS), feedback loop design, fine-tuning at scale, distributed training | Statistical significance testing, A/B testing | `01_concepts/evaluation_metrics.md` |

---

## Which Interview Module to Study First

Choose your path based on your role and background.

```
Your Role?
     │
     ├─ ML Engineer / Data Scientist
     │  └─► Start with: Embeddings → Chunking → Vector DBs
     │      Then: Naive RAG (02_interview_bank/01) → Advanced RAG
     │      Finally: Self-RAG, Adaptive RAG for depth
     │
     ├─ Backend / Systems Engineer
     │  └─► Start with: Vector DBs → System Design Principles
     │      Then: Advanced RAG → Modular RAG
     │      Finally: Agentic RAG, Cost optimization
     │
     └─ Applied Scientist / Research-focused
        └─► Start with: Retrieval Strategies → Evaluation Metrics
            Then: Modular RAG → Agentic RAG
            Finally: Self-RAG, Graph RAG, Structured RAG
```

---

## Common Failure Modes by Maturity Stage

Each stage has signature failure modes that interviewers test for. Knowing them helps you prepare answers.

| Stage | Most Common Interview Gotcha | What Interviewers Are Testing |
|-------|------------------------------|-------------------------------|
| 0–1 | "Why did retrieval miss this relevant document?" | Whether you understand embedding quality and chunking trade-offs |
| 1–2 | "How do you handle a query that needs multiple hops?" | Whether you know when naive retrieval bottlenecks and how to architect beyond it |
| 2–3 | "Your system is slow and expensive. Optimize it." | Whether you can reason about cost centers (embedding, vector storage, LLM) and latency distribution |
| 3–4 | "This query type is still failing. How do you know?" | Whether you have observability and feedback loops to diagnose failures |
| 4+ | "How would you deploy this across teams / regions?" | Whether you understand scaling, consistency, and multi-tenancy |

---

## Production Readiness Checklist

Use this checklist to assess whether your RAG system is ready for production deployment. This is the "meta-interview" checklist: at senior levels, you're not asked about chunking, you're asked why every item here matters.

**Ingestion Pipeline**
- [ ] Incremental indexing works (new documents don't require full re-index)
- [ ] TTL-based eviction or versioning is in place (old documents don't pollute retrieval)
- [ ] Error handling: malformed documents don't crash the pipeline
- [ ] Audit trail: you can track which documents were indexed when and by whom

**Retrieval Performance**
- [ ] P95 latency for a retrieval call is documented and under SLA (typically <200ms)
- [ ] Chunking strategy is calibrated on a labeled probe set (recall@5 measured)
- [ ] Embedding model quality is baselined (MTEB or domain-specific NDCG@5)
- [ ] Reranking is A/B tested or justified as unnecessary

**Generation Quality**
- [ ] Hallucination rate is measured (at least one evaluation metric from RAGAS)
- [ ] Prompt template is hardened against prompt injection (XML-delimited context)
- [ ] LLM token budget is tracked per query to prevent runaway costs
- [ ] Output grading: at least one mechanism to detect and flag bad generations (regex, LLM-as-judge, user feedback)

**Evaluation Infrastructure**
- [ ] A labeled probe set (at least 50 representative queries) exists for regression testing
- [ ] Retrieval and generation metrics are computed on every PR/deployment
- [ ] Metric thresholds are set (e.g., "block deployment if NDCG@5 drops >2%")
- [ ] Production metrics are live and monitored (latency, error rates, feedback signals)

**Operations & Maintenance**
- [ ] Graceful degradation is implemented: if embedding service fails, system degrades gracefully
- [ ] Circuit breakers and timeouts are in place for all external dependencies
- [ ] Monitoring and alerting is configured for all critical components
- [ ] Runbooks exist for common failure scenarios (embedding service down, vector DB out of memory, etc.)

---

## Key Papers and Their Contributions

The RAG field moves fast. These papers define the landscape. Each shaped the architectures in `02_interview_bank/`.

> Full citations with arXiv/DOI links for these and every other paper referenced in this repo: [REFERENCES.md](../REFERENCES.md)

| Year | Paper | What It Introduced | Still Relevant? | Where to Use This Knowledge |
|------|-------|-------------------|-----------------|------------------------------|
| 2020 | Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" | The original RAG framework: retrieve-then-generate | Yes, foundation | `02_interview_bank/01-naive-rag.md` |
| 2023 | Gao et al., "Retrieval-Augmented Generation for Large Language Models: A Survey" | Taxonomy of retrieval strategies (dense, sparse, hybrid) | Yes, reference | `01_concepts/retrieval_strategies.md` |
| 2023 | Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique for Self-Improvement" | First widely-adopted self-improving architecture | Yes, state-of-the-art | `02_interview_bank/07-self-rag.md` |
| 2023 | Jiang et al., "FLARE: Forward-Looking Active REtrieval Augmented Generation" | Iterative retrieval: generate, detect uncertainty, re-retrieve | Emerging, research-frontier | `01_concepts/retrieval_strategies.md` (multi-hop) |
| 2024 | Yan et al., "Corrective Retrieval Augmented Generation" | Validation loop: assess retrieved context, re-retrieve if needed | Yes, production-grade | `02_interview_bank/06-corrective-rag.md` |
| 2024 | Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" | Knowledge graphs + community detection for retrieval | Emerging, enterprise | `02_interview_bank/05-graph-rag.md` |
| 2023 | Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation" | LLM-based evaluation without gold labels | Yes, standard now | `01_concepts/evaluation_metrics.md` |
| 2021 | Formal et al., "SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval" | Learned sparse retrieval — the modern half of hybrid (dense + sparse) search | Yes, best practice | `01_concepts/retrieval_strategies.md` (hybrid) |

---

## Using This Roadmap

1. **Find your level**: Assess your current depth with the skill matrix above.
2. **Plan your progression**: Use the forcing functions to decide whether you need depth at this level or should move to the next.
3. **Prepare interview answers**: Study the failure modes for your target level — interviewers almost always test against them.
4. **Cross-reference**: Use the "Where to Use" column to jump to the relevant Q&A file once you understand the concept.
