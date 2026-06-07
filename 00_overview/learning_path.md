# Learning Path: A Sequenced Curriculum for RAG Mastery

> A sequenced curriculum for mastering RAG — from fundamentals to research-frontier architectures.

---

## Three Learning Tracks

Choose your track based on your available time and interview target.

| Track | Time Budget | Recommended For | Where to Start |
|-------|-------------|-----------------|-----------------|
| **Breadth-First** | 1–2 weeks | Phone screens, quick prep, need to know all 12 types at high level | `00_overview/rag_taxonomy.md`, then each file in `02_interview_bank/` Q1–Q3 only |
| **Depth-First** | 4 weeks | Technical interviews, system design rounds, deep expertise on 3–4 types | `01_concepts/` in full, then `02_interview_bank/` 01–04 in depth |
| **Production-First** | 3 weeks | Senior/staff interviews, operations focus, observability and failure modes | `00_overview/system_design_principles.md`, cost analysis (Q11 across all files), evaluation/observability |

---

## Week-by-Week Study Plan (4-Week Depth-First)

### Week 1: Foundations
**Goal:** Understand the building blocks all RAG systems rest on.

| Day | Topic | Read | Questions to Answer |
|-----|-------|------|-------------------|
| 1–2 | Embeddings | `01_concepts/embeddings.md` full | Can you explain cosine similarity? What embedding model fails on domain-specific text? |
| 2–3 | Chunking Strategies | `01_concepts/chunking_strategies.md` full | Why does chunk size matter? When do you use parent-child chunking? |
| 4 | Vector Databases | `01_concepts/vector_databases.md` full | What's the difference between HNSW and IVF? When do you choose Qdrant vs. Pinecone? |
| 5 | Practice Q&A | Q1–Q3 from `02_interview_bank/01-naive-rag.md` | Answer cold without notes. Practice explaining to a peer. |

**Code Exercise:** Build a minimal retrieval pipeline: embed a 100-document corpus, store in FAISS, retrieve top-5 for 10 test queries, measure recall. Write down chunk size and embedding model choice and why.

---

### Week 2: Core Architectures
**Goal:** Master Naive, Advanced, Modular, and Corrective RAG. Understand the retrieval problem deeply.

| Day | Topic | Read | Questions to Answer |
|-----|-------|------|-------------------|
| 1–2 | Retrieval Strategies | `01_concepts/retrieval_strategies.md` full | What's RRF? When does hybrid retrieval beat dense? What is HyDE? |
| 2–3 | Reranking | `01_concepts/reranking.md` full | Why use a cross-encoder after dense retrieval? What's the latency cost? |
| 4–5 | Naive + Advanced RAG Q&A | Q1–Q10 from `02_interview_bank/01-naive-rag.md` and `02-advanced-rag.md` | Answer all 20 questions cold. These are your baseline |
| 5 | Corrective RAG | `02_interview_bank/06-corrective-rag.md` Q1–Q5 | How does validation change retrieval? When does correction help? |

**Code Exercise:** Implement RRF hybrid retrieval (BM25 + dense). Benchmark hybrid vs. pure dense on 10 test queries. Document the recall gain.

---

### Week 3: Advanced & Specialized Architectures
**Goal:** Understand when to use Agentic, Self-RAG, Graph, Structured, Multimodal, and Long-Context RAG.

| Day | Topic | Read | Questions to Answer |
|-----|-------|------|-------------------|
| 1–2 | Evaluation Metrics | `01_concepts/evaluation_metrics.md` full | What does NDCG@5 mean? What is RAGAS faithfulness? |
| 2–3 | Agentic RAG | `02_interview_bank/04-agentic-rag.md` Q1–Q8 | How does an agent decide to re-retrieve? What fails with agents? |
| 3–4 | Self-RAG + Graph RAG | `02_interview_bank/07-self-rag.md` Q1–Q5, `05-graph-rag.md` Q1–Q5 | Why does Self-RAG require fine-tuning? What does knowledge graph retrieval gain you? |
| 4–5 | Structured + Multimodal RAG | `02_interview_bank/12-structured-rag.md` Q1–Q5, `09-multimodal-rag.md` Q1–Q5 | When do you route to SQL vs. text? How do you embed images? |
| 5 | Long-Context RAG | `02_interview_bank/10-long-context-rag.md` Q1–Q5 | What breaks with context windows >10K tokens? When is long-context better than retrieval? |

**Code Exercise:** Write a retrieval strategy selector: given a query, decide whether to use dense, hybrid, graph-based, or SQL-based retrieval. Justify each choice.

---

### Week 4: System Design & Security
**Goal:** Integrate knowledge into production-grade systems. Understand cost, observability, and security.

| Day | Topic | Read | Questions to Answer |
|-----|-------|------|-------------------|
| 1–2 | System Design Principles | `00_overview/system_design_principles.md` full | What are the five properties of a production RAG? How do you design for cost? |
| 2 | Security & Prompt Injection | `01_concepts/prompt_injection_risks.md` full | What is indirect prompt injection? How do you mitigate it? |
| 3–4 | Cost & Observability Q&A | Q11 from each of `02_interview_bank/01–05.md` (5 cost questions), Q12 from the same files (5 security questions) | Can you optimize cost while maintaining recall? Can you detect a prompt injection attack? |
| 4–5 | Adaptive RAG | `02_interview_bank/11-adaptive-rag.md` Q1–Q10 | When does adaptive routing beat static retrieval? How do you implement routing? |
| 5 | Integration & Practice | Re-read `00_overview/rag_taxonomy.md` and `system_design_principles.md` | Can you draw the canonical system design answer from memory? |

**Code Exercise:** Design a RAG system end-to-end: define the 5 properties, draw the pipeline, compute cost for 10K QPS, instrument for observability, mitigate prompt injection. Write as if you're explaining to a senior engineer.

---

## Concept Dependency Graph

This DAG shows prerequisites for each concept. Follow the arrows: if you want to understand X, make sure you understand everything pointing to it first.

```
Embeddings  ──┐
             ├──► Vector Databases ──┐
Chunking ────┤                       ├──► Naive RAG ──┐
             └──────────────────────►│               │
                                     ├──► Advanced RAG ──┐
Retrieval Strategies ───────────────►│                   │
                                     └──► Modular RAG ───┤
Reranking ──────────────────────────►│                   │
                                     └──► Corrective RAG │
Evaluation Metrics ──────────────────┤                   │
                                     ├──► Agentic RAG ───┤
System Design Principles ────────────┤                   ├──► System Design Round
                                     ├──► Self-RAG ──────┤
Prompt Injection Risks ──────────────┤                   │
                                     ├──► Graph RAG ─────┤
                                     │                   │
                                     ├──► Structured RAG ┤
                                     │                   │
                                     └──► Long-Context RAG┘
```

Note: No circular dependencies exist. This is a strict partial ordering — you can start at the top and work downward.

---

## Depth Calibration by Interview Type

Different interview types test different depth. Use this to calibrate your study.

| Interview Type | Expected Depth | Which Files to Focus On | Time to Spend | Example Question |
|---|---|---|---|---|
| Phone Screen (30–45 min) | Breadth (taxonomy) | `00_overview/rag_taxonomy.md`, Q1–Q3 from each of `02_interview_bank/01–03.md` | 3–5 hours | "What's the difference between Naive and Advanced RAG?" |
| Technical Round (60 min, coding) | Medium (mechanism + code) | `01_concepts/` (all), Q1–Q8 from `02_interview_bank/01–04.md` | 1 week | "Implement hybrid retrieval with RRF" |
| System Design (60 min) | Deep (architecture + trade-offs) | `00_overview/system_design_principles.md`, Q9–Q12 from all `02_interview_bank/` files | 2 weeks | "Design a RAG system for document search. Requirements: 10M docs, <300ms P95 latency." |
| ML Systems Design (90 min) | Deep + research (evaluation + feedback loops) | `01_concepts/evaluation_metrics.md`, `02_interview_bank/07-self-rag.md`, all Q11–Q12 | 2–3 weeks | "How would you build a self-improving RAG system?" |
| Take-Home Project (5–8 hours) | Deep + implementation | All `01_concepts/`, all `02_interview_bank/` | Intensive week | "Build a RAG system for a domain-specific corpus. Measure and optimize retrieval quality." |

---

## Self-Assessment Questions

Answer these without notes. If you get stuck, find the answer in the files and re-read that section. These questions span multiple files — forcing cross-topic synthesis.

1. **Embeddings + Chunking:** Your system retrieves documents but misses relevant ones. How do you diagnose whether it's an embedding problem or a chunking problem?

2. **Vector DBs + Retrieval:** You're evaluating Qdrant vs. FAISS. What's the trade-off? Which would you choose for a 100M vector corpus and why?

3. **Retrieval + Reranking:** Your dense retrieval has 40% recall@5, but after reranking, it's 42%. Is reranking worth the latency cost?

4. **Evaluation + Agentic:** You measure NDCG@5 = 0.7 and faithfulness = 0.85. What does this tell you about your Agentic RAG system?

5. **Chunking + Reranking:** Parent-child chunking vs. fixed-size chunking with reranking — which scales better and why?

6. **Embeddings + Graph RAG:** Your system uses both text embeddings and a knowledge graph. How do you choose which retrieval source to use for a given query?

7. **System Design + Cost:** At 10K QPS with OpenAI embedding and GPT-4, what's your primary cost center? How do you optimize?

8. **Adaptive RAG + Evaluation:** You build an Adaptive RAG that routes queries. How do you measure whether the router improves over Naive RAG?

9. **Reranking + Prompt Injection:** A reranker is supposed to filter out irrelevant documents. Can a reranker be tricked by prompt injection attacks?

10. **Retrieval Strategies + Evaluation:** You're choosing between BM25 and dense retrieval for a medical question-answering system. How do you decide? What metrics inform the choice?

11. **Long-Context + System Design:** With Claude's 200K context window, should you use retrieval or long-context RAG? What's the trade-off?

12. **Self-RAG + Observability:** Your Self-RAG system fine-tunes a model to improve. How do you monitor whether the fine-tuned model is actually better in production?

13. **Prompt Injection + Agentic RAG:** An agent retrieves a document that contains prompt injection. What happens? How do you prevent it?

14. **Structured RAG + Evaluation:** You route some queries to SQL and some to text retrieval. How do you evaluate the system as a whole (one metric across both paths)?

15. **Graph RAG + Chunking:** In Graph RAG, how do you chunk documents when building the knowledge graph? Does chunking strategy affect graph quality?

---

## Resources Beyond This Repo

These resources fill gaps this repo intentionally leaves. Use them for context.

| Resource | Type | What It Covers | Why Use It |
|---|---|---|---|
| [LlamaIndex Documentation](https://docs.llamaindex.ai/) | Code + Tutorials | Production patterns for ingestion, retrieval, evaluation | This repo is concept-heavy; LlamaIndex is implementation-heavy. Use together. |
| [Langchain Handbook](https://www.deeplearning.ai/short-courses/) (DeepLearning.AI) | Course | End-to-end RAG systems with code | Complements the Q&A format with narrative tutorials. |
| [BEIR Benchmark Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) | Benchmark Dataset | Retrieval evaluation across 18 datasets | Test your understanding of retrieval metrics against real benchmarks. |
| [OpenAI Prompt Injection Docs](https://platform.openai.com/docs/guides/prompt-injection-mitigation) | Official Guidance | Prompt injection patterns and defenses | Official best practice; more current than papers. |
| Banerjee et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (2023) | Paper | RAGAS framework in-depth | Deep dive if you want to implement custom evaluation. |

---

## Recommended Study Order

If you're time-constrained, follow this priority order:

1. **Must-know (4 hours):** `01_concepts/embeddings.md` + `01_concepts/chunking_strategies.md`
2. **Essential (6 hours):** `01_concepts/vector_databases.md` + `01_concepts/retrieval_strategies.md`
3. **High-value (4 hours):** `02_interview_bank/01-naive-rag.md` Q1–Q5 + `02-advanced-rag.md` Q1–Q5
4. **Differentiator (4 hours):** `00_overview/system_design_principles.md`
5. **Advanced (6 hours):** `01_concepts/evaluation_metrics.md` + `01_concepts/reranking.md`
6. **Everything else:** Research-frontier (Agentic, Self-RAG, Graph) and specialized (Multimodal, Long-Context) — nice-to-have for senior rounds
