# 19 — Iterative / Multi-hop RAG

> Interleaves retrieval and reasoning across multiple rounds — each generated reasoning step issues a new retrieval, accumulating evidence until a stopping criterion is met — enabling answers that require chaining facts no single passage contains.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
User Query
  → Query Decomposer (splits into sub-questions or seeds the first hop)
  → ┌─────────────── Iterative Loop ───────────────┐
    │  Retriever (retrieves on current hop's query)  │
    │       ↓                                        │
    │  Reasoner (LLM reasons over accumulated         │
    │       evidence, emits next CoT step / sub-Q)    │
    │       ↓                                        │
    │  Stopping-Criterion Controller                  │
    │       (answer-marker? max-hops? no-new-docs?)   │
    │       ↓ (loop if not satisfied)                 │
    └─────────────────────────────────────────────────┘
  → Evidence Accumulator (deduped passages + resolved facts across hops)
  → Generator (synthesizes final answer from accumulated evidence, with provenance)
```

### Key Components

| Component | Responsibility |
|---|---|
| Query Decomposer | Breaks the original question into an initial retrieval query or sub-question plan |
| Retriever | Executes retrieval for the current hop's query against the index |
| Reasoner | LLM reasons over retrieved evidence and generates the next hop's query or CoT step |
| Loop / Stopping Controller | Decides whether to continue hopping (answer marker, max-hops cap, no-new-info check) |
| Evidence Accumulator | Dedupes and carries forward passages/facts across hops, pruning to control context growth |
| Generator | Produces the final answer from the full accumulated evidence chain, with per-hop provenance |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Prompting patterns | IRCoT, Self-Ask, ITER-RETGEN |
| Loop orchestration | LangGraph, custom retrieve-reason-loop controllers |
| Retrieval | Standard vector DB (Qdrant, Pinecone, Weaviate) + BM25 hybrid |
| Verification (optional) | Corrective-RAG-style per-hop evidence checks |
| Evaluation | HotpotQA, 2WikiMultiHopQA, MuSiQue harnesses for per-hop recall |

---

## Q1. What is Iterative / Multi-hop RAG and why is single-shot retrieval insufficient? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Iterative (multi-hop) RAG** runs retrieval more than once per query, where each retrieval is conditioned on what was generated or retrieved in the previous round.

**Why single-shot RAG fails on multi-hop questions:**

Consider: *"What is the capital of the country where the Eiffel Tower's designer was born?"*

- The answer requires three linked facts: (1) Eiffel Tower → designer (Gustave Eiffel), (2) Gustave Eiffel → birth country (France), (3) France → capital (Paris).
- A single embedding of the full question retrieves passages about the Eiffel Tower, but the passage stating "Paris is the capital of France" shares almost no lexical or semantic overlap with the original query — it never surfaces.

**The iterative fix:**

```
Hop 1: "Who designed the Eiffel Tower?"        → Gustave Eiffel
Hop 2: "Where was Gustave Eiffel born?"          → France
Hop 3: "What is the capital of France?"          → Paris  ← answerable now
```

Each hop's retrieval query is generated from the accumulated reasoning, so the system follows the reasoning chain instead of trying to match the whole question at once.

**Distinction from Adaptive RAG (pattern 11):** Adaptive RAG's *classifier* decides whether a query needs multi-hop and routes to it; Iterative RAG is the architecture that actually *executes* the retrieve→reason→retrieve loop.

</details>

---

## Q2. Compare IRCoT, Self-Ask, and ITER-RETGEN. `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

All three interleave retrieval with reasoning, but differ in *what drives the next retrieval*:

| Method | Mechanism | Next-hop query is... | Stopping signal |
|---|---|---|---|
| **IRCoT** (Trivedi 2023) | Interleave Chain-of-Thought with retrieval. Generate one CoT sentence → retrieve on it → append → repeat | The latest CoT sentence | CoT emits "the answer is..." or max hops |
| **Self-Ask** (Press 2022) | Model explicitly asks itself follow-up sub-questions | The generated follow-up sub-question | Model decides no more follow-ups needed |
| **ITER-RETGEN** (Shao 2023) | Use the *full generated answer* from round N as the retrieval query for round N+1 | The previous round's complete answer | Fixed number of iterations (e.g., 3) |

**Key contrasts:**
- **IRCoT** retrieves on partial reasoning (fine-grained, more retrieval calls).
- **Self-Ask** produces an explicit, auditable decomposition (easy to inspect; brittle if the model asks a bad sub-question).
- **ITER-RETGEN** is the simplest — "generation-augmented retrieval": the draft answer is a better retrieval query than the raw question because it contains intermediate entities. Fixed-iteration, so latency is predictable.

**Practical takeaway:** ITER-RETGEN is the easiest to implement (no special prompting structure, fixed loop); IRCoT typically gives the best recall on hard multi-hop benchmarks (HotpotQA, 2WikiMultiHopQA, MuSiQue).

</details>

---

## Q3. Walk through the IRCoT loop end-to-end. `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

IRCoT (Interleaving Retrieval with Chain-of-Thought) alternates a **"reason" step** and a **"retrieve" step**:

```
Initialize: retrieve top-k for the original question → context C

Loop:
  1. Reason: prompt LLM with (question + C + CoT-so-far)
            → generate ONE next CoT sentence
  2. Check: does the sentence contain "answer is" (or max hops hit)?
            → if yes, extract answer, STOP
  3. Retrieve: use that CoT sentence as a query → top-k passages
            → add new passages to C (dedup)
  4. Go to 1
```

**Code skeleton:**

```python
def ircot(question, retriever, llm, max_hops=4, k=4):
    context = retriever.search(question, k=k)
    cot = ""
    for hop in range(max_hops):
        sentence = llm.generate_next_cot_sentence(
            question=question, context=context, cot_so_far=cot
        )
        cot += " " + sentence
        if "answer is" in sentence.lower():
            return extract_answer(sentence), cot
        # Retrieve using the freshly generated reasoning step
        new_docs = retriever.search(sentence, k=k)
        context = dedup(context + new_docs)
    return llm.generate_final_answer(question, context, cot), cot
```

**Why retrieve on the CoT sentence instead of the question?** The CoT sentence names the *current* intermediate entity ("Gustave Eiffel was born in France"), which is the right retrieval signal for the next hop — the original question never mentions France.

</details>

---

## Q4. What is the error-accumulation problem in multi-hop RAG, and how do you mitigate it? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Error accumulation** (a.k.a. cascading / compounding errors): a mistake in an early hop propagates and is amplified in later hops, because each hop conditions on the (possibly wrong) output of the previous one.

If each hop is 90% reliable, a 3-hop chain is only 0.9³ ≈ **73%** reliable — and a wrong hop 1 sends every subsequent retrieval down the wrong path.

**Failure pattern:**
```
Hop 1 retrieves wrong designer ("Stephen Sauvestre") instead of Gustave Eiffel
  → Hop 2 retrieves Sauvestre's birthplace
  → Hop 3 answers the capital of the wrong country
  → Confident, fluent, completely wrong answer.
```

**Mitigations:**

1. **Verify each hop** — add a Corrective-RAG-style check after each retrieval: "Does the retrieved evidence actually support this reasoning step?" Re-retrieve or backtrack if not.
2. **Beam / multiple chains** — explore several reasoning paths in parallel and pick the one with the strongest evidence support (reduces commitment to one bad early hop).
3. **Retrieve-then-decompose vs decompose-then-retrieve** — generating the full decomposition first lets you sanity-check the plan before spending retrievals.
4. **Confidence-gated stopping** — stop and fall back to "I don't have enough information" when hop confidence drops, rather than fabricating the next hop.
5. **Provenance tracking** — keep each hop's supporting passage so a wrong chain is debuggable and the final answer is auditable.

</details>

---

## Q5. How do you decide when to stop iterating? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A multi-hop loop needs an explicit stopping criterion or it loops forever / wastes cost. Common strategies, often combined:

| Criterion | How it works | Risk |
|---|---|---|
| **Answer-token signal** | Stop when the model emits "the answer is …" / a final-answer marker | Model declares done prematurely |
| **Max-hops cap** | Hard limit (e.g., 4–6 hops) | Truncates legitimately deep chains |
| **No-new-information** | Stop when a hop retrieves only already-seen passages | Near-duplicate passages defeat it |
| **Sufficiency check** | A small LLM/classifier judges "can the question be answered now?" each hop | Extra call per hop |
| **Confidence threshold** | Stop when answer log-prob / self-rated confidence exceeds threshold | Confidence is poorly calibrated |

**Recommended combination (production):**
```python
stop = (
    answer_marker_present(sentence)        # primary
    or hop >= MAX_HOPS                     # safety cap (always set)
    or no_new_docs(new_docs, seen)         # progress check
)
```

Always set a **hard max-hops cap** as a backstop — it bounds worst-case latency and cost regardless of model behavior.

</details>

---

## Q6. How does latency and cost scale with hops, and how do you control it? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Multi-hop is fundamentally **sequential** — hop N+1 needs hop N's output — so latency and cost grow roughly **linearly in the number of hops**, and the LLM calls cannot be parallelized across hops.

```
Per hop: 1 reasoning LLM call (300–800ms) + 1 retrieval (50–150ms)
3-hop query ≈ 3 × (~600ms LLM + ~100ms retrieval) ≈ 2.1s
vs. single-shot RAG ≈ 0.7s
```

**Cost driver:** the reasoning LLM calls (context grows each hop as passages accumulate), not retrieval.

**Control strategies:**
1. **Adaptive activation** — gate multi-hop behind a complexity classifier (Adaptive RAG); send simple queries through single-shot. Most traffic is single-hop.
2. **Small model for hops** — use a fast model (Haiku/8B) for intermediate reasoning + query generation; reserve a frontier model for the final synthesis.
3. **Context pruning** — don't carry every passage forward; rerank and keep only top-k relevant to the current reasoning state (also fights context-window growth).
4. **Cap hops aggressively** — empirically, >4 hops rarely helps on standard benchmarks; the long tail of questions needing 5+ hops is small.
5. **Cache hop sub-queries** — common intermediate sub-questions ("capital of France") hit a semantic cache.

</details>

---

## Q7. How does Iterative RAG differ from Agentic RAG and Self-RAG? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

All three involve multiple retrievals, but they differ in *control* and *training*:

| | Iterative / Multi-hop RAG | Agentic RAG | Self-RAG |
|---|---|---|---|
| **Control of the loop** | Fixed retrieve→reason→retrieve schedule | LLM agent freely chooses tools/actions | Model emits special **reflection tokens** |
| **Tools** | Retriever only | Multiple tools (search, calculators, APIs) | Retriever only |
| **Training** | None — prompting only | None — prompting/orchestration | **Fine-tuned** to emit retrieve/critique tokens |
| **Stopping** | Explicit criterion (markers, max hops) | Agent decides "task complete" | Model decides via reflection tokens |
| **Best for** | Compositional factual questions (multi-hop QA) | Open-ended tasks needing diverse tools | High-accuracy domains where you can fine-tune |

**Mental model:**
- **Iterative RAG** = a *fixed pipeline* that loops retrieval and reasoning. Predictable, no training.
- **Agentic RAG** = Iterative RAG generalized to *arbitrary tools and free control flow*. More capable, less predictable.
- **Self-RAG** = the loop control is *learned into the model weights* rather than orchestrated externally.

Iterative RAG is essentially the special case of Agentic RAG where the only tool is the retriever and the control flow is a fixed loop.

</details>

---

## Q8. How do you evaluate a multi-hop RAG system beyond final-answer accuracy? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Final-answer accuracy alone hides *how* the system got there — a system can get the right answer via a wrong chain (lucky) or the wrong answer via a good chain (one bad hop). Evaluate the chain, not just the endpoint.

**Metrics:**

| Dimension | Metric | What it catches |
|---|---|---|
| **Answer** | Exact Match / F1 (HotpotQA, MuSiQue, 2WikiMultiHopQA) | Final correctness |
| **Supporting facts** | Supporting-fact F1 (did it retrieve the gold supporting passages?) | "Right answer, wrong/no evidence" |
| **Per-hop recall** | Recall@k at each hop | Which hop the chain breaks at |
| **Faithfulness** | Each reasoning step entailed by its retrieved evidence | Hallucinated intermediate steps |
| **Hop efficiency** | Avg hops to answer vs. gold hop count | Over/under-retrieving |

**Datasets designed for this:** HotpotQA (2-hop + supporting-fact supervision), 2WikiMultiHopQA, MuSiQue (2–4 hops, built to resist shortcut/single-hop solutions), Bamboogle.

**Diagnostic move:** compute per-hop recall to find the breaking hop — if hop 1 recall is high but hop 2 collapses, the problem is query generation between hops, not the retriever itself.

</details>

---

## Q9. What is the difference between "decompose-then-retrieve" and "retrieve-then-reason" iterative strategies? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Two structural approaches to multi-hop:

**1. Decompose-then-retrieve (plan-first)**
```
LLM decomposes question → [sub-q1, sub-q2, sub-q3]  (up front)
Retrieve + answer each sub-question
Compose final answer from sub-answers
```
- ✅ The plan is inspectable before any retrieval; independent sub-questions can be retrieved **in parallel**.
- ❌ The decomposition is made *blind* — without seeing intermediate results, so it can't adapt (it may not know sub-q3 depends on sub-q2's answer).

**2. Retrieve-then-reason (interleaved, e.g., IRCoT)**
```
Retrieve → reason one step → retrieve on that step → ... (adaptive)
```
- ✅ Each hop adapts to what was just found; handles **dependent** chains where you can't know hop 3 until you've done hop 2.
- ❌ Strictly sequential (no parallelism); more retrieval calls; error accumulation.

**When to use which:**
- **Independent decomposition** ("Compare the GDP of France and Japan") → decompose-then-retrieve, run sub-queries in parallel.
- **Dependent / bridge chains** ("capital of the country where X was born") → interleaved, because each hop's query depends on the prior answer.

Hybrid systems classify the question type first, then pick the strategy.

</details>

---

## Q10. Design an iterative RAG system for a financial-research analyst assistant. `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

```
USE CASE: Analyst asks "Which of Acme's top-3 suppliers has the highest
          debt-to-equity ratio, and how exposed is Acme to it?"
CORPUS: 10-K/10-Q filings, earnings calls, internal supplier DB

This is inherently multi-hop:
  Hop 1: Acme's top-3 suppliers          (from Acme filings / supplier DB)
  Hop 2: D/E ratio of each supplier      (from each supplier's filings)
  Hop 3: Acme's exposure to the riskiest (from Acme filings + contracts)

PIPELINE
────────
1. Query classifier (Adaptive RAG gate):
     single-hop ("What was Acme's Q3 revenue?") → standard RAG, skip loop
     multi-hop  (this query)                    → iterative path

2. Decompose-vs-interleave router:
     Hop 1 → Hop 2 is INDEPENDENT across suppliers → parallel sub-retrieval
     Hop 2 → Hop 3 is DEPENDENT (need the riskiest) → sequential

3. Per-hop retrieval = hybrid (BM25 + dense) over filings, filtered by
   metadata {company, fiscal_period, doc_type} to avoid cross-company bleed

4. Per-hop VERIFICATION (Corrective step):
     numeric facts (D/E ratio) re-checked against the structured DB (Structured RAG)
     — financial numbers must never be hallucinated

5. Stopping: sufficiency check + MAX_HOPS=5 cap

6. Final synthesis (frontier model) with FULL provenance:
     every figure cites {filing, page, fiscal period}

GUARDRAILS
──────────
- Metadata filtering prevents the classic multi-hop bug: retrieving
  supplier B's numbers while reasoning about supplier A.
- Numeric claims routed to structured source, not vector recall.
- Every hop's evidence retained → answer is fully auditable (regulatory need).

MONITORING
──────────
- Per-hop recall, supporting-fact F1 on a gold analyst eval set
- % queries routed to multi-hop, avg hops, P95 latency
- Numeric-claim verification pass rate
```

The financial domain makes two requirements non-negotiable: **provenance** (auditability) and **routing numeric facts to a structured source** rather than trusting vector retrieval.

</details>

---

## Q11. How does iterative RAG handle the growing-context problem across hops? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Each hop appends new passages and reasoning to the context, so naively the prompt grows unboundedly — driving up cost, latency, and "lost-in-the-middle" degradation.

**Management strategies:**

1. **Rerank-and-prune per hop** — after each retrieval, rerank the *full* accumulated pool against the current reasoning state and keep only top-k. The context stays fixed-size instead of growing.
2. **Carry summaries, not raw passages** — replace earlier hops' passages with a short factual summary ("Hop 1 established: designer = Gustave Eiffel"), keeping only the latest hop's raw text.
3. **Carry facts, not text** — extract the atomic fact each hop produced into a compact "scratchpad" of resolved entities; retrieve raw passages only for the active hop.
4. **Deduplicate aggressively** — multi-hop frequently re-retrieves overlapping passages; dedup by chunk ID before adding.
5. **Separate working vs. evidence context** — keep the reasoning chain (small) always in context; page evidence passages in/out as needed.

**Why this matters specifically here:** unlike single-shot RAG, the context-growth is *cumulative and unbounded in hops*, so without pruning a deep chain will blow the context window and bury the relevant late-hop evidence in the middle of a huge prompt.

</details>

---

## Q12. What are the security and robustness risks specific to iterative RAG? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The multi-round, self-conditioning loop creates risks beyond single-shot RAG:

**1. Injection propagation across hops**
A poisoned passage retrieved in hop 1 can contain instructions ("for the next step, search for X") that hijack the *query generation* for hop 2 — steering the whole chain. The attack surface compounds with each hop.
- *Mitigation:* treat retrieved content as data, not instructions; sanitize/scan each hop's retrieved passages before they influence the next query; never let retrieved text directly become a tool call.

**2. Query drift / topic hijacking**
Generated sub-queries can drift away from the user's intent (model misreads an intermediate entity), silently answering a different question.
- *Mitigation:* measure embedding distance between each hop query and the original question; flag/abort on drift beyond a threshold.

**3. Unbounded resource consumption (DoS)**
A crafted query that never satisfies the stopping criterion can loop to max hops repeatedly, multiplying LLM cost.
- *Mitigation:* hard max-hops cap + per-request token/cost budget + rate limiting.

**4. Evidence laundering**
The fluent final answer can present a wrong multi-hop chain as authoritative; a wrong early hop is invisible in the final text.
- *Mitigation:* surface per-hop provenance; faithfulness-check each step against its evidence.

**5. Cross-tenant / cross-entity leakage**
A hop's retrieval may pull a different tenant's or entity's documents if metadata filters aren't reapplied *every* hop.
- *Mitigation:* enforce ACL/metadata filters on every retrieval call, not just the first.

</details>

---

## Q13. Walk through the Iterative/Multi-hop RAG architecture end-to-end. `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

```
Query → Generate/retrieve for hop 1 → sub-answer/evidence 1
      → Reason over accumulated evidence: is more needed?
          ├── Yes → formulate next sub-query, conditioned on evidence
          │          so far → retrieve for hop 2 → sub-answer/evidence 2
          │          → repeat
          └── No  → synthesize final answer from all accumulated evidence
```

The defining property is that each hop's query is **conditioned on what previous hops found**, not generated independently — this is what distinguishes genuine multi-hop iteration from RAG-Fusion's (#18) parallel, independent reformulations (#18 Q15's own comparison draws this same line): a multi-hop question like "what is the birthplace of the director of the highest-grossing 2019 film" cannot be answered by any single reformulation of the original query, since the second retrieval (the director's birthplace) literally cannot be formulated until the first retrieval (who directed the film) has already returned a name.

</details>

---

## Q14. What is the research origin of iterative multi-hop RAG? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

IRCoT (Trivedi et al., *Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions*, arXiv:2212.10509, 2022) established interleaving chain-of-thought generation with retrieval calls — generate one reasoning sentence, retrieve based on it, append the result, generate the next sentence — as a prompted, training-free technique on a frozen LLM. Self-Ask (Press et al., *Measuring and Narrowing the Compositionality Gap in Language Models*, arXiv:2210.03350, 2022) took a related but distinct approach, explicitly prompting the model to decompose a compound question into a sequence of self-posed sub-questions, each answered (with retrieval) before the next is posed.

Both papers share the same underlying insight this file's Q1 describes: many real questions require information that only becomes knowable after an earlier piece of information is retrieved, which no single-shot retrieval — however well the query is rewritten or expanded (RAG-Fusion, #18; HyDE, #22) — can address, since those techniques all still retrieve exactly once based on the original query alone.

</details>

---

## Q15. How does Iterative Multi-hop RAG compare to HippoRAG's (#20) single-pass multi-hop retrieval? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Both target genuine multi-hop questions, but HippoRAG (#20) answers them in a **single retrieval pass** — a Personalized PageRank computation that traverses a pre-built graph's multi-hop connections in one shot, with no LLM-mediated intermediate reasoning steps at query time. Iterative Multi-hop RAG instead chains **several separate retrieval-and-reasoning rounds**, with each round's query formulated by the LLM based on what previous rounds found — the "hops" happen as sequential, LLM-driven retrieval calls rather than as paths through a pre-built graph structure.

The trade-off (already drawn from HippoRAG's own side, #20 Q13-Q14): HippoRAG's single-pass approach is cheaper and faster per query, but depends on having pre-built a graph that already encodes the relevant multi-hop connections; Iterative Multi-hop RAG needs no such pre-built structure — it can chain retrieval over a completely flat, ungraphed corpus — at the cost of paying for several sequential LLM calls per query and being exposed to the error-accumulation risk (Q4) that a single-pass graph traversal doesn't have in the same way.

</details>

---

## Q16. What is the single distinctive mechanism that separates Iterative RAG from single-shot RAG? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The distinctive mechanism is **conditioning each retrieval query on the accumulated results of previous retrieval steps**, rather than retrieving once, upfront, based only on the original query. Single-shot RAG (Naive, #01, and every enhancement built on it that still retrieves exactly once — reranking, hybrid search, even RAG-Fusion's #18 parallel reformulations) can only ever search for what the original query's text implies is needed; it has no mechanism to discover, mid-process, that a different piece of information is actually required first before the real question can even be formulated.

This single capability is what makes genuine multi-hop questions answerable at all without a pre-built graph structure (contrast with HippoRAG, #20, Q15) — the cost is that each additional hop requires its own LLM reasoning call and its own retrieval call, which is why hop count directly drives both latency and cost (Q6) in a way no single-shot architecture's tuning knobs can replicate.

</details>

---

## Q17. What are the key tuning knobs for an iterative RAG loop, and how do you choose them? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Knob | Effect | Starting point |
|---|---|---|
| Max hops | Caps how many sequential retrieval rounds a query can trigger | 3-5, tuned to your domain's genuine multi-hop depth; always enforce a hard ceiling regardless of the stopping criterion's quality |
| Stopping-criterion signal (Q5) | Determines how confidently the loop recognizes "I have enough to answer" | Explicit, example-driven stopping instructions outperform a vague "stop when done" prompt, the same lesson learned for agentic loops generally (#04 Q18) |
| Per-hop `k` (chunks retrieved per hop) | More chunks per hop improve that hop's own recall but add to the growing context each subsequent hop has to process (Q11) | 3-5 per hop is typical; smaller than a single-shot system's typical k, since multiple hops compound total retrieved volume |
| Evidence-carrying strategy across hops (full accumulated context vs. condensed summary, Q11) | Full context preserves detail but grows unboundedly; condensation bounds growth at the risk of losing detail | Condense older hops' evidence into a running summary once accumulated context approaches a size threshold, mirroring conversational memory's own tiering approach (#21 Q17) |

Max hops and per-hop `k` interact multiplicatively on total retrieval volume and cost (Q6) — the same multiplicative-cost pattern seen in every other multi-step architecture in this bank (RQ-RAG's #51 branch count times sub-query count, ToT-RAG's #37 branching factor times depth), making both worth tuning together against a cost budget rather than independently maximizing each for accuracy.

</details>

---

## Q18. How do you evaluate whether each additional hop is actually contributing to answer quality? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Beyond final-answer accuracy (Q8), measure accuracy as a function of hop count directly: run the same multi-hop evaluation set with the loop artificially capped at 1 hop, 2 hops, 3 hops, etc., and plot accuracy against hop count — the expected pattern is a rising curve that plateaus once the loop has genuinely gathered enough evidence, with any hops beyond the plateau point adding cost (Q6) without a corresponding accuracy gain.

This curve reveals two distinct things: where your stopping criterion (Q5, Q17) *should* be terminating for a representative question (informing whether it's currently over- or under-hopping), and whether your evaluation set's questions actually need as many hops as your architecture assumes — a plateau reached earlier than your configured max hops suggests either the stopping criterion is under-triggering (running unnecessary extra hops) or your query distribution is less genuinely multi-hop than the architecture was built to handle, both of which point at cost being wasted somewhere in the pipeline.

</details>

---

## Q19. What is the characteristic failure mode when a hop retrieves a plausible but wrong intermediate fact? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Because each hop's query is conditioned on the previous hop's result (Q16), a wrong intermediate fact doesn't just affect that one hop's own accuracy — it corrupts every subsequent hop's query formulation, since those queries are built assuming the wrong fact is correct. This is the specific mechanism behind Q4's error-accumulation problem: a plausible-looking wrong answer at hop 1 ("the director of the film is X" when it's actually Y) causes hop 2 to search for "X's birthplace" — a well-formed, successfully-executed retrieval that returns a confident, correct-looking answer to entirely the wrong question, with nothing in the pipeline flagging that anything went wrong, since every individual hop "succeeded" on its own terms.

**Detection:** for a labeled multi-hop evaluation set with known-correct intermediate facts (not just final answers), check each hop's intermediate result against ground truth specifically — a wrong final answer with a correct chain of intermediate facts points at a synthesis problem in the final step, while a wrong final answer traceable to one specific wrong intermediate hop points at this exact failure mode, and these two require completely different fixes. **Mitigation:** apply a lightweight consistency or confidence check at each hop before committing to it as the basis for the next hop's query (surfacing per-hop provenance and confidence, per this file's own security-and-robustness mitigation), rather than only validating the fully-assembled final answer, since by the final-answer stage an early wrong hop has already propagated too far to trace back cheaply.

</details>

---

## Q20. What are the limitations of Iterative Multi-hop RAG, and how might the field evolve? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Current limitations: (1) **error accumulation compounds silently** (Q4, Q19) — a wrong intermediate fact corrupts every subsequent hop with no built-in mechanism to notice; (2) **cost scales multiplicatively with hop count** (Q6, Q17) — the most expensive per-query architecture among this bank's foundational tier when hop counts run high; (3) **stopping-criterion quality directly gates both cost and completeness** (Q5, Q18) — a poorly-tuned criterion either wastes cost on unnecessary hops or terminates before genuinely gathering enough evidence; (4) **growing context across hops requires active management** (Q11) — unbounded accumulation risks the same lost-in-the-middle degradation flagged for long-context architectures generally (#10 Q14).

Likely evolution: this exact set of limitations is precisely what motivated the newer architectures elsewhere in this bank built specifically to address one or more of them — HippoRAG's (#20) single-pass PPR retrieval eliminates sequential hop cost entirely for corpora that can support a pre-built graph; CoRAG (#50) and Search-R1 (#42) replace hand-prompted iteration with a trained policy specifically optimized to avoid wasted or drifting hops; and Auto-RAG/DeepRAG (#49) add per-step retrieve-or-reason decisions on top of the same iterative shape to skip hops the model can answer parametrically. A mature production system increasingly treats "iterative multi-hop RAG" as this file's foundational, prompting-based baseline — the reference point every subsequent, more specialized multi-hop architecture in this bank is built to improve on in one specific dimension (cost, robustness, or efficiency) rather than the final answer to genuinely multi-hop retrieval.

</details>

---

## Real-World Applications

| Application | Domain | Why Iterative / Multi-hop RAG Fits |
|---|---|---|
| Complex question answering (HotpotQA-style assistants) | Search / Knowledge | Bridge questions require chaining facts across documents that no single passage contains |
| Financial & competitive research analysts | Finance / Enterprise | "Find the riskiest supplier and our exposure" decomposes into dependent retrieval hops with auditable provenance |
| Scientific literature synthesis | Research / Biomed | Tracing a mechanism (gene → pathway → disease → drug) needs sequential, evidence-linked hops |
| Legal case & precedent research | Legal | Following citation chains and statutory cross-references is inherently multi-hop |
| Troubleshooting & root-cause assistants | DevOps / Support | Diagnosis chains (symptom → component → dependency → known issue) adapt each hop to the prior finding |
