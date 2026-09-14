# 51 — RQ-RAG (Refinement-aware Query for RAG)

> Fine-tunes the LLM itself to explicitly rewrite, decompose, or disambiguate a query — as a learned skill invoked via special tokens — rather than relying on a frozen model prompted with a fixed rewriting template.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
User Query
    │
    ▼
Fine-tuned Refinement Policy (LLM emits a special action token)
    │
    ├── <rewrite>       → produces a single reformulated query
    ├── <decompose>     → produces N sub-queries
    ├── <disambiguate>  → produces a clarified, specific query
    └── <no-op>         → uses the query as-is
    │
    ▼
Retriever (runs one retrieval call PER refined query)
    │
    ▼
Tree of Candidate Paths (each path = one refinement choice + its retrieved evidence + a draft answer)
    │
    ▼
PPL-based Path Selector (scores each path by output token perplexity / confidence)
    │
    ▼
Final Answer (from the highest-scoring path in the decoding tree)
```

### Key Components

| Component | Responsibility |
|---|---|
| Refinement-tuned LLM | Single model fine-tuned to output one of REWRITE / DECOMPOSE / DISAMBIGUATE / no-op via special control tokens, instead of a hand-written rewriting prompt |
| Query Refiner | Executes the chosen operation, producing one rewritten query, several decomposed sub-queries, or one disambiguated query |
| Retriever | Standard dense/sparse retriever, invoked once per refined query produced |
| Tree Decoder | Expands multiple candidate refinement paths in parallel (since refinement is stochastic/multi-choice at each step) |
| Path Selector | Picks the best final answer across candidate paths using a perplexity-based confidence score |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Base model | Llama2-7B (as used in the original paper), any open-weight instruction-tuned LLM |
| Fine-tuning data construction | Self-instruct-style generation of (query, refinement-type, refined query) triples, distilled from a stronger LLM |
| Fine-tuning framework | Hugging Face `transformers` + `trl` / `peft` (LoRA) for supervised fine-tuning on refinement traces |
| Retriever | Any dense retriever (FAISS, Elasticsearch BM25, ColBERT) — RQ-RAG is retriever-agnostic |
| Reference implementation | [chanchimin/RQ-RAG on GitHub](https://github.com/chanchimin/RQ-RAG) |

---

## Q1. What is RQ-RAG and how does it differ from the query-rewriting techniques (HyDE, Multi-Query, Step-Back) already used in Advanced RAG? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**RQ-RAG** (Refinement-aware Query for RAG, from *"RQ-RAG: Learning to Refine Queries for Retrieval Augmented Generation"*, Chan et al., 2024, [arXiv:2404.00610](https://arxiv.org/abs/2404.00610)) **fine-tunes** an LLM so that query refinement becomes a *learned, explicit skill* the model performs on its own — rather than a behavior induced by a frozen model plus a clever prompt.

**02-advanced-rag.md already covers three prompted refinement tricks:**

| Technique | How it works | Model state |
|---|---|---|
| HyDE | Prompt the LLM to hallucinate a fake answer, embed that instead of the query | Frozen, zero-shot prompted |
| Multi-Query expansion | Prompt the LLM to generate N paraphrases of the query | Frozen, zero-shot prompted |
| Step-Back prompting | Prompt the LLM to generate a more abstract/general version of the query | Frozen, zero-shot prompted |

**RQ-RAG's difference:**

```
Prompted rewriting (HyDE / Multi-Query / Step-Back):
  Frozen LLM + hand-written instruction
       │
       ▼
  "Please rephrase this query for better retrieval..."
       │
       ▼
  One fixed strategy applied uniformly to every query

RQ-RAG:
  Fine-tuned LLM has LEARNED, from training data, when to:
       │
       ├─ REWRITE       (query is fine but phrased badly for retrieval)
       ├─ DECOMPOSE      (query bundles multiple sub-questions)
       ├─ DISAMBIGUATE   (query is ambiguous / underspecified)
       └─ pass through unchanged (query is already retrieval-ready)
       │
       ▼
  The model itself DECIDES which operation fits THIS query,
  and can chain several refinement steps adaptively.
```

**Key distinction:** prompted rewriting techniques apply *one* strategy to *every* query via prompt engineering on a frozen model. RQ-RAG fine-tunes the model to **select among multiple refinement operations per-query** and to **chain them**, closer to how a human researcher would first clarify an ambiguous question, then split it into parts, then reword each part for a search engine.

</details>

---

## Q2. What are the three refinement operations, and how are they represented at inference time? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

RQ-RAG introduces three explicit refinement types, each triggered by a special control token the fine-tuned model learns to emit:

| Operation | Special token | Purpose | Example |
|---|---|---|---|
| **Rewrite** | `<rewrite>` | Rephrase a poorly-worded query into retrieval-friendly language | "that movie with the guy who lost his memory" → "Christopher Nolan Memento plot" |
| **Decompose** | `<decompose>` | Split a compound/multi-hop query into independent sub-queries | "Compare the GDP of France and its population growth rate" → ["What is France's GDP?", "What is France's population growth rate?"] |
| **Disambiguate** | `<disambiguate>` | Resolve an underspecified query into a concrete one | "What is the capital?" → "What is the capital of Australia?" (using conversation context) |

**Inference-time control flow (simplified):**

```python
def rq_rag_step(query: str, context: str = "") -> list[str]:
    """
    The fine-tuned model is prompted to first emit a refinement-type token,
    then the refined query/queries conditioned on that choice.
    """
    prompt = f"""{context}
Query: {query}
Choose a refinement action: <rewrite>, <decompose>, <disambiguate>, or <answer_directly>."""

    action = refinement_llm.generate(prompt, max_tokens=16)   # e.g. "<decompose>"

    if action == "<decompose>":
        sub_queries = refinement_llm.generate(
            f"{prompt}\n<decompose>\nSub-queries:", max_tokens=128
        )
        return parse_list(sub_queries)          # ["query A", "query B", ...]

    elif action == "<rewrite>":
        rewritten = refinement_llm.generate(f"{prompt}\n<rewrite>\nRewritten query:")
        return [rewritten]

    elif action == "<disambiguate>":
        clarified = refinement_llm.generate(f"{prompt}\n<disambiguate>\nClarified query:")
        return [clarified]

    return [query]   # <answer_directly> — no refinement needed
```

**Why special tokens instead of a free-form instruction?** Because the model was **fine-tuned** on traces containing these tokens, emitting `<decompose>` reliably switches the model into a *decomposition-conditioned generation mode* it has practiced — this is far more reliable at inference time than hoping a frozen model correctly interprets an ad hoc natural-language instruction like "decompose this if needed."

</details>

---

## Q3. How is the RQ-RAG training data constructed, and how is the model fine-tuned? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

RQ-RAG needs **supervised traces** that pair a query with the correct refinement type and the correct refined output — this data doesn't exist naturally, so the paper synthesizes it using a stronger teacher LLM.

**Step 1 — Synthesize refinement traces:**

```python
TEACHER_PROMPT = """Given the query below, decide whether it needs to be:
(a) rewritten for better search-engine retrieval,
(b) decomposed into independent sub-questions, or
(c) disambiguated using the given context.
If none apply, say so.

Query: {query}
Context: {context}

Output format:
Action: <rewrite|decompose|disambiguate|none>
Refined: <the refined query or list of sub-queries>"""

# Run over a large pool of QA queries (e.g. from ambiguous-QA, multi-hop QA,
# and search-log-style datasets) using a strong teacher model (e.g. GPT-4-class)
# to generate (query, action, refined_query) triples.
```

**Step 2 — Filter with retrieval-and-answer feedback:**

A generated refinement is only kept if using it to retrieve documents actually **improves** downstream answer correctness versus using the raw query — this turns the dataset into a curated set of refinements that are empirically useful, not just plausible-looking.

**Step 3 — Supervised fine-tuning:**

```python
# Each training example is a full trace: query → action token → refined query
# → retrieved docs → answer, concatenated as one sequence, with loss computed
# only on the tokens the model must generate (action token + refined query + answer).

training_example = """Query: Compare the GDP of France and its population growth rate.
<decompose>
Sub-queries: ["What is France's GDP?", "What is France's population growth rate?"]
[... retrieved passages ...]
Answer: France's GDP is approximately $3.0T; its population growth rate is approximately 0.2% annually."""

# Standard causal-LM fine-tuning (full fine-tune or LoRA) on a base Llama2-7B checkpoint,
# using cross-entropy loss over the full trace.
```

**Step 4 — Multi-path tree decoding at inference:**

Because the same query could plausibly warrant different refinement types, RQ-RAG doesn't commit to a single greedy path — it expands a **decoding tree** with several candidate refinement branches (e.g. try both `<rewrite>` and `<decompose>`), retrieves for each, drafts an answer per branch, and then picks the best branch using a perplexity-based confidence score (see Q4).

</details>

---

## Q4. How does the tree-decoding / path-selection mechanism decide which refined query to actually use? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Because refinement is a **stochastic decision** the model makes per query, RQ-RAG doesn't rely on a single greedy refinement — it explores multiple candidate paths and scores them.

```
                         Original Query
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                   ▼
       <rewrite>          <decompose>        <disambiguate>
            │                  │                   │
     retrieve(rewritten)  retrieve(sub-q1)    retrieve(clarified)
            │             retrieve(sub-q2)          │
            ▼                  ▼                   ▼
       draft answer A     draft answer B       draft answer C
            │                  │                   │
            └──────────────────┼───────────────────┘
                                ▼
                    Score each path by output
                    perplexity / confidence
                                │
                                ▼
                    Select lowest-perplexity
                    (highest-confidence) path
                                │
                                ▼
                          Final Answer
```

```python
def rq_rag_tree_decode(query: str, context: str, k_branches: int = 3) -> str:
    candidate_paths = []

    for action in ["<rewrite>", "<decompose>", "<disambiguate>"]:
        refined_queries = refine(query, context, action)          # from Q2
        docs = [retriever.search(q) for q in refined_queries]
        answer, logprobs = generate_answer_with_logprobs(query, docs)

        # Confidence proxy: average per-token log-probability of the answer
        confidence = sum(logprobs) / len(logprobs)
        candidate_paths.append({"action": action, "answer": answer, "confidence": confidence})

    best = max(candidate_paths, key=lambda p: p["confidence"])
    return best["answer"]
```

**Why perplexity/confidence as the selector, and not another LLM judge?**
- It's essentially free — the generation model already computes token log-probabilities during decoding, no extra model call needed
- Empirically, answers grounded in *correctly refined* retrieval tend to have lower perplexity because the retrieved evidence is more directly relevant, making the answer more "predictable" given the context
- This keeps the extra latency of tree decoding bounded — the added cost is running the retriever + generator per branch, not an extra verification pass

**Multi-hop benefit:** for a query needing 2 hops, the `<decompose>` branch retrieves for each sub-query independently, giving the generator focused evidence for each hop rather than one retrieval call diluted across a compound question — this is the core reason RQ-RAG's reported gains are larger on multi-hop QA datasets than single-hop ones.

</details>

---

## Q5. What are the costs and failure modes of RQ-RAG compared to prompted rewriting, and when would you NOT use it? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

RQ-RAG trades **inference-time cost and engineering overhead** for **higher-quality, per-query-adaptive refinement**. That trade isn't always worth it.

**Costs:**

| Cost dimension | Prompted rewriting (HyDE/Multi-Query/Step-Back) | RQ-RAG |
|---|---|---|
| Training required | None — works with any frozen instruction-tuned LLM | Yes — requires building a refinement-trace dataset and fine-tuning |
| Inference latency | 1–2 extra LLM calls | Multiple branches × (refine + retrieve + generate) per query — tree decoding multiplies retrieval and generation calls by the number of branches explored |
| Maintainability | Swap prompt anytime, no retraining | Refinement behavior baked into weights — updating strategy requires re-fine-tuning |
| Portability across base models | Works with any LLM via prompting | Fine-tune is tied to a specific base model checkpoint; upgrading the base model means re-running the fine-tuning pipeline |

**Failure modes:**

1. **Distribution shift in refinement decisions** — if production queries look very different from the synthetic refinement-trace training data, the model may pick the wrong action (e.g. decomposing a query that was actually fine as-is), adding latency without benefit.
2. **Compounding retrieval calls** — `<decompose>` on a query with many implicit sub-questions can trigger a retrieval call per sub-query per branch; without a cap, this can multiply retrieval cost badly on adversarial or overly compound queries.
3. **Confidence-score miscalibration** — perplexity-based path selection can favor a fluent-but-wrong answer over a correct-but-awkwardly-phrased one, since perplexity measures predictability, not factual correctness.

**When to prefer prompted rewriting instead:**
- Rapid prototyping or low query volume, where the fine-tuning investment doesn't pay off
- Frequently swapping base models (e.g. evaluating multiple vendor LLMs) — prompted techniques port instantly, fine-tuned refinement doesn't
- Cost-sensitive deployments where the multi-branch tree-decoding overhead isn't justified by the accuracy gain

**When RQ-RAG is worth it:**
- High query volume with a stable base model, where the one-time fine-tuning cost amortizes
- Query workloads with a genuine mix of ambiguous, compound, and well-formed queries — a single hand-written prompt strategy can't adapt per-query the way a fine-tuned action-selection policy can
- Can be combined with **Adaptive RAG** (file 11) — the adaptive router decides *whether* to retrieve at all, while RQ-RAG decides *how* to refine the query once retrieval is triggered; the two are complementary routing layers, not competitors

</details>

---

## Q6. Walk through the RQ-RAG architecture end-to-end. `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

```
User Query
    │
    ▼
Fine-tuned Refinement Policy (LLM emits a special action token)
    ├── <rewrite>       → produces a single reformulated query
    ├── <decompose>     → produces N sub-queries
    ├── <disambiguate>  → produces a clarified, specific query
    └── <no-op>         → uses the query as-is
    │
    ▼
Retriever (runs one retrieval call PER refined query)
    │
    ▼
Tree of Candidate Paths (each path = refinement choice + evidence + draft answer)
    │
    ▼
PPL-based Path Selector (scores each path by output token perplexity/confidence)
    │
    ▼
Final Answer (from the highest-scoring path)
```

Every stage after the special-token decision exists to hedge against that decision being uncertain: rather than committing to one refinement choice, the tree decoder (Q4) explores several plausible operations in parallel and lets a cheap, free confidence signal (token perplexity, no extra model call) pick the winner after the fact. This is what distinguishes RQ-RAG from a system that just always applies one fixed refinement strategy — the model both *learns which operation fits a given query* and *hedges against getting that choice wrong* by exploring alternatives.

</details>

---

## Q7. What is the single distinctive mechanism that separates RQ-RAG from prompted query-rewriting techniques? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The distinctive mechanism is a **fine-tuned, per-query action-selection policy expressed via special control tokens**, replacing a single fixed prompted strategy applied uniformly to every query. HyDE, Multi-Query expansion, and Step-Back prompting (Q1) each apply one hand-chosen transformation to every query a frozen model sees; RQ-RAG's fine-tuned model instead decides, per query, *which* of several operations (rewrite, decompose, disambiguate, or none) actually fits that specific query's needs — a compound question gets decomposed, an ambiguous one gets disambiguated, a poorly-phrased-but-otherwise-fine one gets rewritten, and an already-good query passes through untouched.

This per-query adaptivity is only possible because the operation choice is a *learned* skill (Q3's fine-tuning on synthesized traces) rather than a *prompted* one — a frozen model told "decompose if needed, otherwise rewrite" in one prompt has no comparably reliable, trained-in mechanism for making that judgment call consistently across a diverse query distribution the way a model specifically fine-tuned on labeled (query, correct-action) pairs does.

</details>

---

## Q8. How does RQ-RAG compare to CoRAG (#50)? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Both fine-tune a model to make better retrieval-query decisions rather than relying on prompting, but differ in what's being learned and how. RQ-RAG fine-tunes a **discrete choice among a small, named set of refinement operations** (rewrite/decompose/disambiguate) via special tokens — the training data comes from a teacher-model-synthesized and outcome-filtered dataset (Q3), and the model's output is individually interpretable (you can see which operation it chose). CoRAG (#50) fine-tunes **free-form, multi-hop query reformulation within an evolving reasoning chain**, with training data constructed via rejection sampling over complete chains rather than labeled individual operations, and no constrained operation vocabulary at all.

The practical distinction mirrors DeepRAG's (#49) structured-vs-free-form trade-off: RQ-RAG's fixed operation vocabulary is easier to audit and reason about (you always know which of three named things happened) but can't express reformulation patterns outside that vocabulary; CoRAG's unconstrained reformulation can adapt to arbitrary multi-hop patterns but offers no equivalent interpretability into *why* a given reformulation was chosen. Both use tree/multi-path decoding with a selection step at inference (RQ-RAG's perplexity-based path selector, Q4; CoRAG's best-of-N reranking, #50 Q13) to hedge against any single sampled choice being wrong.

</details>

---

## Q9. What is the research origin of RQ-RAG, and what does its training data pipeline look like at a high level? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

RQ-RAG was introduced by Chan et al., *RQ-RAG: Learning to Refine Queries for Retrieval Augmented Generation* (arXiv:2404.00610, 2024), fine-tuning a Llama2-7B base model on synthesized refinement traces generated by a stronger teacher LLM. The training pipeline (detailed in Q3) is itself a notable methodological contribution: rather than hand-labeling which refinement operation fits each training query, the paper uses a teacher model to propose an action and a refined query, then **filters by outcome** — a synthesized refinement is only kept if actually using it to retrieve improves downstream answer correctness versus using the raw query, converting a subjective "does this look like a good rewrite" judgment into an empirically-measured one.

This outcome-filtering step is conceptually similar to CoRAG's rejection sampling (#50 Q2) — both convert final-answer correctness into a training-data quality filter rather than relying on hand-labeled intermediate supervision — applied here to individual refinement operations rather than complete multi-hop chains. The paper's reported gains concentrate on multi-hop QA specifically (Q4), consistent with the `<decompose>` operation's core value proposition: giving the generator focused, per-sub-question evidence rather than one retrieval call diluted across a compound question.

</details>

---

## Q10. What are the key tuning knobs for RQ-RAG, and how do you choose them? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Knob | Effect | Starting point |
|---|---|---|
| `k_branches` (tree decoding width, Q4) | More branches explored per query improve the odds the best refinement is among the candidates, but multiply retrieval and generation cost | 3 (one per operation type), per Q4's pseudocode; consider dropping unlikely operations early via a cheap upfront classifier to reduce branch count |
| Teacher model choice (Q3's data synthesis) | A stronger teacher produces higher-quality synthesized refinement traces, directly setting a ceiling on what the fine-tuned student can learn | Use the strongest model you can afford for data synthesis — this is a one-time offline cost, unlike inference-time model choice |
| Outcome-filtering strictness (Q3 step 2) | Stricter filtering (larger required improvement margin over the raw query) yields cleaner but smaller training data | Calibrate against how much training data volume you need vs. how much noise the fine-tuning process can tolerate |
| Base model choice for fine-tuning | Determines inference cost and quality ceiling; also determines how portable the trained refinement policy is if the base model changes later | Llama2-7B in the original paper; any open-weight instruction-tuned model works, sized to your latency/cost budget |

`k_branches` is the most operationally significant knob because it directly multiplies inference cost (Q16) — unlike CoRAG's adaptive chain length (#50 Q3), which can shrink cost for easy queries, RQ-RAG's default tree decoding explores a fixed branch set regardless of how confident an upfront read of the query might already be about which operation fits, making branch-pruning heuristics (Q13's filtering logic applied at inference rather than just training time) a natural cost-control extension.

</details>

---

## Q11. How do you evaluate RQ-RAG's action-selection quality against prompted rewriting? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Beyond final-answer accuracy (the paper's headline comparison against HyDE/Multi-Query/Step-Back baselines, Q1), evaluate **action-selection accuracy directly**: build a labeled set of (query, correct-action) pairs — including queries that genuinely need no refinement at all — and measure whether the fine-tuned policy's chosen action (before tree decoding even runs) matches the labeled correct action. This decomposes "does RQ-RAG produce good final answers" from "does the action-selection policy actually pick the right operation," since tree decoding's multi-branch hedge (Q4) can mask a poorly-calibrated action-selection policy by exploring the correct operation anyway as one of several branches — a decomposition-quality issue that would only surface if you specifically measured selection accuracy, not just final-answer accuracy after the safety net of exploring all branches.

Segment by query category (already-well-formed, compound/multi-hop, ambiguous/context-dependent) to check whether the policy's accuracy is uniform or concentrated in one category — the paper's own strongest reported gains are on multi-hop QA (Q9), so a production evaluation should confirm this pattern holds for your query distribution rather than assuming uniform improvement across all query types, mirroring the same segmented-evaluation discipline used throughout this bank (e.g., LongRAG's #45 Q11).

</details>

---

## Q12. What is the characteristic failure mode of action-selection distribution shift, and how do you detect it in production? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Q5 flags this as a training-data mismatch risk: if production queries look meaningfully different from the synthetic refinement-trace training data (Q3) — different domain vocabulary, different typical query length or structure, different ambiguity patterns — the fine-tuned policy can systematically pick the wrong action for query types under-represented in training, adding tree-decoding latency and cost without the accuracy benefit the correctly-chosen operation would have provided.

**Symptom:** an elevated rate of `<decompose>` or `<disambiguate>` selections on queries that a human reviewer would judge as already well-formed (or vice versa — a `<no-op>` selection on genuinely compound queries) is the direct signature, distinguishable from a general accuracy problem because it's specifically about the *action choice* rather than the *final answer quality* (Q11's decomposition matters here again). **Detection:** periodically sample production queries and their selected actions for human review, comparing action-selection accuracy over time — since this file's own recommendation is to fine-tune on synthesized traces resembling expected production query patterns, a rising mismatch rate over time (as production traffic naturally evolves away from what the training data represented) is the concrete signal that a refresh of the training-trace dataset (Q3) — synthesizing new traces from current production query samples — is due, rather than assuming the original training data remains representative indefinitely.

</details>

---

## Q13. How do you implement and tune the retrieval-and-answer feedback filtering step in training data construction? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Q3's step 2 filters teacher-synthesized refinements by whether they actually improve downstream answer correctness — this requires running a full retrieve-and-answer pass for both the raw query and the refined query, comparing outcomes, and keeping only refinements that measurably help:

```python
def filter_refinement_by_outcome(query: str, refined_queries: list[str], gold_answer: str,
                                   retriever, answer_model, min_improvement: float = 0.0) -> bool:
    """Keep a synthesized refinement only if it improves answer correctness
    over using the raw query, per Q3 step 2."""
    raw_docs = retriever.search(query, k=5)
    raw_answer = answer_model.generate(query, raw_docs)
    raw_score = score_answer(raw_answer, gold_answer)  # e.g. F1 or EM

    refined_docs = [retriever.search(q, k=5) for q in refined_queries]
    flat_docs = [d for docs in refined_docs for d in docs]
    refined_answer = answer_model.generate(query, flat_docs)
    refined_score = score_answer(refined_answer, gold_answer)

    return (refined_score - raw_score) > min_improvement
```

`min_improvement` is the key tuning knob here: a threshold of exactly 0.0 keeps any refinement that helps even marginally, maximizing training data volume at the risk of including noisy, barely-better refinements; a higher threshold produces a smaller but more confidently-useful training set. This filtering step is what elevates RQ-RAG's training data above a naive "ask a teacher model to propose refinements and trust its judgment" approach (Q9) — a teacher model can propose a plausible-sounding but actually-unhelpful refinement, and only the outcome-based filter catches this, exactly the same empirical-validation principle used in CoRAG's rejection sampling (#50 Q2).

</details>

---

## Q14. How does RQ-RAG's special-token operation vocabulary compare to Auto-RAG/DeepRAG's (#49) retrieve-vs-parametric action space? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Both use a small, discrete, named action vocabulary learned via fine-tuning, but the actions serve entirely different purposes. RQ-RAG's actions (rewrite/decompose/disambiguate/no-op) all assume retrieval will happen and decide *how to formulate the query* for it. Auto-RAG/DeepRAG's actions (RETRIEVE/PARAMETRIC, #49 Q3) decide *whether to retrieve at all*, with no query-formulation dimension — a PARAMETRIC decision means the subquery is answered from memory with no retrieval call whatsoever, a choice RQ-RAG's vocabulary doesn't include at all (RQ-RAG's `<no-op>` still means "retrieve using the query as-is," not "skip retrieval").

The two are complementary rather than overlapping: a production system could layer them — first, an Auto-RAG/DeepRAG-style decision on whether a given subquery needs retrieval at all, and only if retrieval is needed, an RQ-RAG-style decision on how to formulate the query for it. This combination would give a system both of these architectures' distinctive efficiency and precision benefits simultaneously — skipping retrieval entirely where parametric knowledge suffices (#49), and formulating a well-refined query specifically when retrieval is genuinely triggered (this file).

</details>

---

## Q15. How would you build a decision-gate benchmark to decide whether RQ-RAG's fine-tuning investment is worth it over prompted rewriting? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Q5's cost table already lays out the qualitative trade-offs; a decision gate quantifies them for your specific deployment:

```
1. Confirm preconditions: a stable base model you don't expect to swap
   frequently, and either an existing labeled QA dataset or the
   willingness to build one via teacher-model synthesis (Q3, Q9) --
   without these, prompted rewriting is likely the only practical option.

2. Baseline: measure HyDE/Multi-Query/Step-Back (whichever prompted
   technique(s) you'd otherwise deploy) on your actual query
   distribution, tracking accuracy, latency, and cost.

3. Build a modest synthesized refinement-trace dataset (Q3, Q13) and
   fine-tune a candidate RQ-RAG model -- a smaller-scale trial run
   before committing to full-scale data synthesis and training.

4. Measure the candidate's action-selection accuracy (Q11) and final-
   answer accuracy against the same baseline query set, plus per-query
   cost/latency accounting for tree decoding's branch multiplication (Q16).

5. Gate: proceed to full deployment only if (a) query volume and query-
   type diversity (a genuine mix of well-formed, compound, and
   ambiguous queries, per Q5) justify per-query adaptive refinement
   over one fixed prompted strategy, AND (b) the measured accuracy
   improvement clears the added tree-decoding cost at your volume,
   AND (c) your base model is stable enough that the fine-tuning
   investment won't need to be redone on a short cycle.
```

The precondition in step 1 is the one most likely to eliminate RQ-RAG from consideration early — teams frequently evaluating or swapping between multiple frontier LLM providers, where "stable base model" doesn't hold, should generally default to prompted rewriting regardless of how favorable Q11's accuracy comparison looks in an isolated trial.

</details>

---

## Q16. What is the cost and latency overhead of tree decoding at scale? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Tree decoding (Q4) multiplies per-query cost by the number of branches explored: each branch requires its own refine-retrieve-generate sequence, so `k_branches=3` means roughly 3x the retrieval calls and 3x the generation calls of a single-path system, before accounting for `<decompose>`'s own sub-query multiplication within its branch (a decompose branch with 3 sub-queries adds 3 retrieval calls just for that one branch). Illustrative worst case per Q5's failure mode 2: a query the model decomposes into many implicit sub-questions, explored as one of several tree branches, can trigger a retrieval-call count that's the product of branch count and sub-query count — without an explicit cap, a single query could generate a surprisingly large number of retrieval calls relative to a standard single-retrieval RAG baseline.

Illustrative cost comparison at 1M queries/month, `k_branches=3`, average 2 sub-queries when `<decompose>` is chosen: roughly 4-5 retrieval calls and 3-4 generation calls per query on average (weighted across which branches actually get explored), versus standard RAG's 1 retrieval + 1 generation call — a meaningful multiple, though bounded and predictable given a fixed branch count, unlike Q5's uncapped-decompose worst case. **Controls:** cap sub-query count per `<decompose>` branch explicitly (never let a single branch's decomposition run away); consider pruning unlikely branches before full exploration via a cheap upfront classifier estimating which 1-2 operations are plausible for a given query, rather than always exploring all 3-4 by default (directly reducing the `k_branches` multiplier for the common case where one operation is clearly most likely); and batch retrieval calls across branches where the underlying retriever supports batched queries, reducing round-trip overhead even when the total query count doesn't decrease.

</details>

---

## Q17. What security and trust risks are specific to a fine-tuned query-refinement policy? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

- **Adversarial queries exploiting `<decompose>` to explode retrieval cost** — Q5's failure mode 2 (compounding retrieval calls) becomes a denial-of-service vector if an attacker can craft queries specifically designed to trigger maximal decomposition (deeply nested compound structure) combined with tree decoding's branch multiplication (Q16), pushing per-query cost far above typical levels; a rate limit on effective sub-query-and-branch count per request is a necessary safeguard distinct from ordinary request-rate limiting, since a single malicious *request* can still generate disproportionate backend load.
- **Training-trace poisoning via the teacher-model synthesis pipeline** — since training data comes from a teacher model's proposals filtered by outcome (Q3, Q13), a systematic bias or blind spot in the teacher model's judgment propagates into the student's learned action-selection policy, similar in kind to the teacher-trace-distillation risk noted for Auto-RAG (#49 Q17) — the student has no independent way to know the teacher's refinement judgment was itself well-calibrated, only that the outcome-filter happened to pass it.
- **Perplexity-based path selection is gameable in principle** — since path selection (Q4) scores candidates purely by output token perplexity, content that happens to make an answer more "predictable" in a shallow sense (generic, hedge-y phrasing that's statistically common) without being more *correct* could be systematically favored over a genuinely correct but less fluently-phrased answer (Q19) — an adversarial or simply low-quality retrieved passage that nudges the generator toward safer-sounding, higher-perplixity-scoring phrasing could win path selection despite being less accurate.
- **Action-token spoofing via prompt injection** — if any part of the query-refinement prompt incorporates untrusted user or document content adjacent to where the model is meant to emit its action token, there's a theoretical risk of injected content influencing which action token gets emitted, similar to the general prompt-injection concern for any structured-output-producing LLM call.

Mitigation: enforce hard caps on sub-query count and total branch-driven retrieval calls per request, independent of what the model itself proposes; audit the teacher-model synthesis pipeline's proposals for systematic bias the same way any distillation-based training data would be audited; and recognize perplexity-based selection's blind spot (fluency over correctness) as a reason to pair it with periodic sampled correctness verification (Q19) rather than trusting it as a complete quality signal.

</details>

---

## Q18. Design an RQ-RAG-based enterprise search system for a jargon-heavy corpus. `[Advanced]` `[Scenario]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Requirements:** employees search using casual, everyday phrasing while the corpus uses internal terminology and acronyms; queries range from simple lookups to compound multi-part questions; maintaining a hand-written synonym dictionary has proven brittle and high-maintenance.

```
1. Training data synthesis (Q3, Q9): generate refinement traces specifically
   using REAL internal search queries paired with the internal
   terminology that should have been used, via a teacher model prompted
   with examples of the casual-to-internal-terminology mapping -- this
   is exactly the domain-specific reformulation-idiom learning
   RQ-RAG's <rewrite> operation is suited for (Real-World Applications,
   this file), replacing the brittle hand-maintained dictionary with a
   learned mapping that generalizes to phrasings not explicitly listed.

2. Action distribution monitoring (Q12): given this corpus's query mix
   skews toward <rewrite> (terminology translation) more than
   <decompose> or <disambiguate>, monitor action-selection distribution
   specifically for drift toward inappropriate decomposition or
   disambiguation on what should be straightforward rewrite cases.

3. Branch pruning (Q16): since <rewrite> is expected to dominate for
   this use case, consider a cheap upfront classifier that skips
   exploring <decompose>/<disambiguate> branches for queries confidently
   classified as simple terminology-translation cases, reducing tree-
   decoding cost for the common case.

4. Outcome-filtered retraining cadence (Q12, Q13): as internal
   terminology evolves (new product names, reorganized team names),
   periodically resynthesize training traces from recent search logs
   and re-fine-tune, treating the refinement policy as needing the same
   ongoing maintenance as a synonym dictionary would have, but updated
   via data rather than manual curation.

5. Evaluation (Q11): track rewrite quality specifically via downstream
   retrieval success (did the rewritten query actually retrieve the
   internally-relevant document), not just fluency of the rewritten
   query text.
```

The key design choice is treating RQ-RAG's `<rewrite>` operation as a learned replacement for the brittle synonym-dictionary approach — the value proposition here isn't RQ-RAG's multi-operation flexibility (this use case is rewrite-dominated) but specifically its ability to learn domain terminology mapping from data rather than requiring ongoing manual dictionary maintenance.

</details>

---

## Q19. What happens when perplexity-based path selection favors a fluent-but-wrong answer, and how do you debug it? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Q5's failure mode 3 identifies the core issue: perplexity measures how predictable an answer's tokens are given its context, not whether the answer is factually correct — an answer phrased in a generic, hedge-heavy, common pattern can score lower perplexity (higher "confidence" by this proxy) than a correct but more specific, less commonly-phrased answer, causing path selection (Q4) to systematically prefer the wrong branch in exactly the cases where correctness and fluency diverge.

**Debugging:** (1) on a labeled evaluation set, measure the correlation between each candidate path's perplexity score and its actual correctness — a weak or inconsistent correlation (rather than the expected "lower perplexity, more often correct" pattern) confirms the selector is behaving more like a fluency detector than a correctness detector for your domain; (2) inspect cases where path selection picked a path that scored well on perplexity but was later confirmed incorrect, checking specifically for patterns in the winning answer's phrasing (generic hedging language, common sentence structures) that might explain why it scored artificially well; (3) if miscalibration is confirmed and material, consider supplementing perplexity with a lightweight, cheap correctness signal — a fast heuristic check (does the answer contain expected entity types, does it directly address the question's specific ask) or, at higher cost, occasional sampled verification via a stronger judge model — rather than relying on perplexity alone, accepting the added latency/cost this introduces specifically for cases the perplexity signal treats as close calls between candidate paths.

</details>

---

## Q20. What are the limitations of RQ-RAG, and how might the field evolve? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Current limitations: (1) **tree decoding's cost scales with branch count regardless of query difficulty** (Q16) — unlike CoRAG's adaptive chain length (#50), RQ-RAG's default exploration doesn't shrink for queries where the right operation is already obvious upfront, without additional branch-pruning engineering; (2) **perplexity-based path selection conflates fluency with correctness** (Q5, Q19) — a structural blind spot in the selection mechanism, not just a calibration issue that better tuning fully resolves; (3) **training data quality is bottlenecked by teacher-model judgment plus outcome filtering** (Q3, Q9, Q17) — both stages can introduce systematic bias that's hard to fully audit; (4) **the fixed three-operation vocabulary may not cover every useful refinement pattern** — a query need that doesn't cleanly map to rewrite/decompose/disambiguate has no expressive slot in the current design, unlike CoRAG's unconstrained reformulation (Q8).

Likely evolution: **adaptive branch pruning** (as sketched in Q16, Q18) becoming a standard component rather than an ad hoc optimization, narrowing the cost gap with more compute-adaptive architectures; **richer or learned correctness signals for path selection** replacing perplexity alone (Q19), potentially incorporating retrieval-quality or entailment-style checks cheaply alongside the free perplexity signal rather than relying on it exclusively; and continued convergence with the broader family of learned-retrieval-decision architectures in this bank (CoRAG, Search-R1, Auto-RAG/DeepRAG) — a plausible unification is a single fine-tuned policy that jointly decides whether to retrieve at all (#49's action space), how to formulate the query if so (this file's operations), and how many hops to chain (CoRAG's #50 chain-length knob), rather than treating these as three separately-trained architectural choices as they exist today.

</details>

---

## Q21. A small consultancy wants a lightweight search tool that rewrites vague client questions into concrete sub-queries. Should they fine-tune an RQ-RAG-style model, or is prompted rewriting good enough here? `[Basic]` `[Scenario]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A handful of consultants asking vague, compound questions against a small internal knowledge base is a reasonable match for RQ-RAG's conceptual value — deciding per-query whether to rewrite, decompose, or disambiguate — but Q15's decision gate points fairly clearly toward prompted rewriting instead at this scale: low query volume doesn't amortize the fine-tuning investment, and a small consultancy evaluating or swapping between vendor LLMs periodically (a common pattern for a firm this size) runs directly into the base-model-coupling cost Q5 and Q7 flag, where a fine-tuned refinement policy is tied to one specific checkpoint and has to be retrained if the underlying model changes.

The practical approach is to approximate RQ-RAG's three operations via prompting on a frozen model: ask it to first classify whether a client question needs rewriting, decomposing into sub-questions, or clarifying against prior conversation context, then execute accordingly — capturing most of the adaptive benefit without a training pipeline, special tokens, or tree-decoding infrastructure.

Revisit the fine-tuning question later, not now: if the consultancy's query volume grows substantially and settles on one stable base model long enough to amortize training, Q15's gate may then pass, particularly if a genuine mix of well-formed, compound, and ambiguous questions is common enough that one fixed prompted strategy starts feeling inadequate — but that's a future decision point, not today's.

</details>

---

## Q22. A multinational's internal search platform has to rewrite and decompose ambiguous employee queries across 15 business units that each use their own jargon. How does RQ-RAG's design need to change to avoid learning just one unit's vocabulary? `[Advanced]` `[Scenario]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Fifteen business units with inconsistent terminology is a harder version of Q18's single-corpus jargon-translation scenario, and the key risk this scale adds is a policy that quietly overfits to whichever business units contributed the most training queries, silently underperforming on the others — training data synthesis (Q3, Q9) has to deliberately sample across all fifteen units' real search logs, not just whichever units were easiest to get data from first, or the fine-tuned model will systematically favor the terminology of its best-represented units.

Action-distribution monitoring (Q12) needs to be segmented per business unit rather than aggregated, since different units will genuinely have different natural action mixes — a unit with heavy internal-acronym use will lean on `<rewrite>` far more than a unit whose questions are mostly already well-formed — and a single global expectation would mask a unit-specific miscalibration that looks fine in the aggregate. Given the scale implies real query volume, branch-pruning (Q16) is worth the engineering effort to keep tree-decoding cost bounded, and the perplexity-based path-selection risk in Q19 is sharper here than usual: a fluent-sounding disambiguation that happens to use the wrong business unit's terminology for an ambiguous term is exactly the kind of confidently-wrong answer perplexity alone won't catch, arguing for a lightweight business-unit classifier as a supplementary signal ahead of path selection.

Retraining cadence (Q12, Q18) needs to track organizational reality, not just calendar time — reorganizations, renamed teams, and merged business units can shift terminology faster than a single stable corporation would, so resynthesizing training traces from recent search logs needs to happen on a schedule tied to organizational change, and per-unit action-selection accuracy should be tracked explicitly so any underrepresented unit's drift is caught rather than averaged away.

</details>

---

## Real-World Applications

- **Multi-hop QA assistants**: Decomposing compound questions (e.g. "How does X compare to Y over time?") into independently retrievable sub-queries, similar in spirit to file 19's Iterative Multi-Hop RAG but driven by a fine-tuned action rather than an iterative loop
- **Conversational search**: The `<disambiguate>` operation resolves pronoun/ellipsis-heavy follow-up queries ("what about the second one?") using prior turn context, an alternative to the memory-augmentation approach in file 21 (Memory/Conversational RAG)
- **Enterprise search over jargon-heavy corpora**: `<rewrite>` learns domain-specific query reformulation (casual phrasing → internal terminology) without needing a hand-maintained synonym dictionary
- **Research assistants**: Chaining decomposition then rewriting lets the system handle "explain the tradeoffs between A and B, citing recent work" as a structured multi-step retrieval plan rather than one noisy retrieval call
