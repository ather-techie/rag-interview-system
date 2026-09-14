# 50 — CoRAG (Chain-of-Retrieval Augmented Generation)

> Trains the model itself — via rejection-sampled retrieval chains — to dynamically reformulate its query at each step based on the evolving reasoning state, and exposes retrieval-chain length as a test-time compute scaling knob.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
Training-time: building retrieval-chain data via rejection sampling
──────────────────────────────────────────────────────────────────
Existing RAG dataset (query → final answer only, no chain labels)
  │
  ▼
Sample MANY candidate retrieve-then-reason chains per question
  (query reformulation → retrieve → sub-answer → reformulate → retrieve → ...)
  │
  ▼
Reject chains that DON'T reach the correct final answer
Keep chains that DO reach the correct final answer
  │
  ▼
Fine-tune model on (query, kept retrieval chain, final answer) triples
  → model learns to autonomously reformulate queries mid-chain


Inference-time: test-time compute scaling
──────────────────────────────────────────────────────────────────
Query
  │
  ▼
┌───────────────── Retrieval Chain (learned, not prompted) ─────────────┐
│  Step 1: reformulate query given current state → retrieve → sub-answer │
│  Step 2: reformulate query given updated state  → retrieve → sub-answer│
│  Step 3: ... (chain length controllable at test time)                  │
└─────────────────────────────────────────────────────────────────────────┘
  │
  ▼
Decoding strategy controls compute:
  - greedy (1 chain, fixed length)
  - best-of-N sampled chains + reranking
  - longer chains for harder queries
  │
  ▼
Final Answer Generator (synthesizes from the full learned retrieval chain)
```

### Key Components

| Component | Responsibility |
|---|---|
| Rejection Sampler (training-time) | Samples many candidate retrieve-reformulate-reason chains per training question, discards ones that don't reach the correct answer |
| Chain-augmented Training Set | (query, kept retrieval chain, final answer) triples used to fine-tune the model |
| Query Reformulator (learned) | Model-internal capability — generates the next retrieval query conditioned on the evolving reasoning state, learned from data rather than prompted |
| Retriever | Executes retrieval for each reformulated query in the chain |
| Test-time Decoding Controller | Chooses chain length / sampling strategy (greedy, best-of-N, chain-length budget) to trade compute for accuracy |
| Final Answer Generator | Synthesizes the answer from the full retrieval chain the model produced |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Reference implementation | CoRAG (Wang, Chen, Yang, Huang, Dou, Wei — Microsoft Research, "Chain-of-Retrieval Augmented Generation," arXiv:2501.14342, Jan 2025; NeurIPS 2025) |
| Training data source | KILT benchmark tasks, augmented with sampled retrieval chains |
| Rejection sampling | Custom sampling + answer-match filtering over many candidate chains per question |
| Base models fine-tuned | Open-source LLMs (e.g. Llama-family) fine-tuned end-to-end on chain data |
| Retriever | Standard dense retriever (E5/Contriever-style) over the KILT corpus |
| Decoding strategies | Greedy chain decoding, best-of-N sampled chains with reranking, dynamic chain-length budgets |

---

## Q1. What is CoRAG, and how does it differ from IRCoT's interleaved retrieval-and-reasoning? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**CoRAG** (Chain-of-Retrieval Augmented Generation — Wang et al., Microsoft Research, arXiv:2501.14342, Jan 2025) trains an "o1-like" RAG model that retrieves and reasons step by step, **dynamically reformulating its query based on the evolving reasoning state**, before producing a final answer.

**IRCoT** (mentioned in file 19, `19-iterative-multihop-rag.md`) achieves interleaved retrieval-and-reasoning purely through **prompting**, on a **frozen, unmodified model**:

```
IRCoT:
  Frozen LLM + fixed prompt template
  → generate one CoT sentence → retrieve based on it → append retrieved text
  → generate next CoT sentence → retrieve → ... → answer
  No training/weight updates. The model's ability to interleave well
  depends entirely on prompt engineering and the base model's existing
  reasoning capability.
```

**CoRAG trains the model itself** to perform this reformulate-retrieve-reason loop, using retrieval chains constructed via rejection sampling as training signal:

```
CoRAG:
  Training: sample many candidate retrieval chains per question,
            keep only chains that reach the correct answer,
            fine-tune the model on those chains.
  Inference: the fine-tuned model NATIVELY reformulates its query at
             each step — this is a learned capability, not a prompted one.
```

**The practical difference:**

| Aspect | IRCoT | CoRAG |
|---|---|---|
| Mechanism | Prompting, frozen model | Fine-tuning on rejection-sampled chains |
| Query reformulation | Implicit, via whatever CoT sentence the frozen model happens to generate | Explicit learned skill, optimized against chains that provably reach correct answers |
| Model requirement | Works out-of-the-box with any capable LLM | Requires a training pipeline and a base model you can fine-tune |
| Test-time compute control | Not a first-class design feature | First-class: chain length / sampling strategy is an explicit scaling knob |

</details>

---

## Q2. How does CoRAG construct training data via rejection sampling when only final answers are labeled? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Most RAG datasets (including KILT tasks) only label the correct final answer — they don't come with ground-truth intermediate retrieval queries or reasoning chains. CoRAG solves this with **rejection sampling**: generate many candidate chains, keep only the ones that actually land on the correct answer.

```python
def sample_candidate_chain(query: str, llm, retriever, max_hops: int = 4) -> dict:
    """Sample one candidate retrieve-reformulate-reason chain."""
    state = {"original_query": query, "steps": []}
    current_query = query

    for hop in range(max_hops):
        # Model proposes next retrieval query given accumulated state
        reformulated = llm.sample_reformulation(state, current_query, temperature=0.8)
        passages = retriever.search(reformulated, k=5)
        sub_answer = llm.sample_subanswer(state, reformulated, passages, temperature=0.8)

        state["steps"].append({
            "query": reformulated, "passages": passages, "sub_answer": sub_answer
        })
        current_query = sub_answer  # condition next reformulation on this

        if llm.sample_stop_decision(state):
            break

    final_answer = llm.sample_final_answer(state)
    return {"chain": state["steps"], "final_answer": final_answer}


def build_corag_training_set(dataset: list[dict], llm, retriever,
                              n_samples: int = 20) -> list[dict]:
    """
    dataset: [{"query": ..., "gold_answer": ...}, ...]
    Returns kept chains: only those whose sampled final_answer matches gold.
    """
    training_examples = []
    for item in dataset:
        for _ in range(n_samples):
            candidate = sample_candidate_chain(item["query"], llm, retriever)
            if answers_match(candidate["final_answer"], item["gold_answer"]):
                training_examples.append({
                    "query": item["query"],
                    "chain": candidate["chain"],
                    "final_answer": candidate["final_answer"],
                })
                # keep multiple accepted chains per question if desired,
                # or just the first/shortest one found
    return training_examples

def answers_match(predicted: str, gold: str) -> bool:
    return predicted.strip().lower() == gold.strip().lower()  # or fuzzy/EM match
```

**Why rejection sampling rather than hand-labeling chains?** Hand-labeling gold retrieval-reformulation chains for every question in a large knowledge-intensive dataset (KILT spans multiple tasks) is infeasible at scale. Rejection sampling instead uses the *existing* answer labels as a filter: any chain sampled from a capable-enough teacher model that happens to arrive at the correct final answer is accepted as a plausible, useful training signal for *how* to get there — even though the chain itself was never hand-verified for perfect step-by-step correctness.

**Fine-tuning:** the target model is then trained (typically via standard supervised fine-tuning) on the kept (query, chain, final_answer) triples, learning to imitate the reformulation-retrieval-reasoning pattern end-to-end.

</details>

---

## Q3. What decoding strategies does CoRAG use at test time to control the retrieval chain, and how do they trade off compute vs. accuracy? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

At inference, CoRAG exposes chain length and sampling strategy as explicit, tunable levers — this is the "test-time compute scaling" analogue to reasoning-model scaling (more inference compute → better accuracy, up to diminishing returns).

```python
def corag_greedy(query: str, model, retriever, chain_length: int) -> str:
    """Cheapest: one deterministic chain of a fixed length."""
    state = {"steps": []}
    current_query = query
    for _ in range(chain_length):
        reformulated = model.reformulate(state, current_query, greedy=True)
        passages = retriever.search(reformulated, k=5)
        sub_answer = model.sub_answer(state, reformulated, passages, greedy=True)
        state["steps"].append({"query": reformulated, "sub_answer": sub_answer})
        current_query = sub_answer
    return model.final_answer(state)


def corag_best_of_n(query: str, model, retriever, chain_length: int, n: int) -> str:
    """More compute: sample N full chains, rerank, pick the best."""
    candidates = []
    for _ in range(n):
        candidates.append(corag_sample_one_chain(query, model, retriever, chain_length))
    return model.rerank_and_select(candidates)


def corag_adaptive_length(query: str, model, retriever, max_length: int = 6) -> str:
    """Let the model decide how many hops it actually needs, up to a cap."""
    state = {"steps": []}
    current_query = query
    for _ in range(max_length):
        reformulated = model.reformulate(state, current_query)
        passages = retriever.search(reformulated, k=5)
        sub_answer = model.sub_answer(state, reformulated, passages)
        state["steps"].append({"query": reformulated, "sub_answer": sub_answer})
        current_query = sub_answer
        if model.is_confident_to_stop(state):
            break
    return model.final_answer(state)
```

**Scaling behavior (as characterized in the paper):**

| Strategy | Compute cost | Accuracy effect |
|---|---|---|
| Greedy, short chain | Lowest | Fine for simple/single-hop queries; underperforms on multi-hop |
| Greedy, long chain | Medium | Better for multi-hop, but wastes compute on simple queries and risks chain drift if reformulation degrades over long chains |
| Best-of-N sampled chains + reranking | Highest | Best accuracy, especially on hard multi-hop tasks — reported >10 point EM improvement over strong baselines on multi-hop QA |
| Adaptive length (model decides to stop) | Query-dependent | Approaches best-of-N accuracy at a fraction of the compute, by only spending extra hops where the question actually needs them |

**Key point:** because chain length/sampling is a *decoding-time* choice, the same trained CoRAG model can be deployed at different latency/cost/accuracy operating points without retraining — analogous to how a reasoning model's "thinking budget" can be dialed up or down at inference time.

</details>

---

## Q4. Why can longer retrieval chains degrade rather than improve accuracy in CoRAG, and how does the paper address it? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Naively, one might expect "more retrieval hops = strictly better accuracy," but CoRAG's chains can drift or accumulate noise the same way any iterative multi-hop process can:

```
Chain drift example:
  Hop 1: reformulated query is accurate, retrieves good passage
  Hop 2: reformulation conditioned on a slightly imprecise sub-answer from hop 1
         → reformulated query drifts off-topic
  Hop 3: retrieves passages for the now off-topic query → irrelevant context
  Hop 4: final answer synthesized from an increasingly noisy chain
```

Each additional hop is generated conditioned on the model's *own* prior sub-answer — if an early sub-answer is subtly wrong, every downstream reformulation compounds that error, similar to the general error-accumulation risk in any iterative/multi-hop retrieval architecture (see file 19, `19-iterative-multihop-rag.md`, on stopping-criterion design for the same class of problem).

**How CoRAG mitigates this:**

- **Rejection sampling only keeps chains that reach the correct final answer** — so during training, the model is never taught to imitate a drifted chain that still happened to be sampled; only chains with a *correct outcome* survive into the training set, which implicitly biases the learned reformulation policy toward corrective, on-track behavior.
- **Best-of-N test-time sampling + reranking** directly compensates for the fact that any single sampled chain can drift — sampling several chains and reranking by (estimated) final-answer quality lets bad individual chains be filtered out at inference time rather than trusted blindly.
- **Adaptive stopping** avoids forcing the model into more hops than the question actually needs, which limits the number of opportunities for drift to occur in the first place.

**Practical implication for tuning in production:** chain length is not a free accuracy dial — past the point where the question's genuine multi-hop depth is satisfied, additional hops mostly add compute cost and drift risk rather than accuracy gains, so adaptive-length decoding tends to outperform a fixed long-chain-for-everything policy.

</details>

---

## Q5. How would you decide between IRCoT-style prompting and training a CoRAG model for a new multi-hop RAG system, and can the two be combined? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

This is fundamentally a build-vs-prompt tradeoff, and the right choice depends on scale, latency budget, and how much training infrastructure is available.

**When IRCoT-style prompting is the better choice:**

| Condition | Reason |
|---|---|
| No fine-tuning infrastructure / using a closed frontier model | IRCoT works with any capable frozen LLM via prompting alone |
| Rapidly evolving requirements | Prompt changes ship instantly; no retraining cycle |
| Low query volume | Doesn't amortize the upfront cost of building a rejection-sampled training set + fine-tuning pipeline |
| Need to swap base models frequently | A prompting scaffold transfers across model versions with minor tuning; a fine-tuned CoRAG checkpoint is tied to the model it was trained on |

**When training a CoRAG-style model is the better choice:**

| Condition | Reason |
|---|---|
| High query volume, latency/cost-sensitive | A model with an internalized reformulation policy needs less per-step prompt overhead and can be decoded greedily far more reliably than a prompted frozen model |
| Need reliable test-time compute scaling | CoRAG's chain-length/best-of-N knobs are trained-in and calibrated against real accuracy gains, rather than an ad hoc prompt hyperparameter |
| Domain-specific query reformulation patterns | Fine-tuning on rejection-sampled chains from your own domain data teaches reformulation behavior specific to your corpus, rather than relying on a frozen model's general-purpose reasoning |
| You control the base model weights | Fine-tuning is only viable with open-weights models (or a provider that supports custom fine-tuning) |

**Combining both:** in practice, IRCoT-style prompting is a reasonable way to *bootstrap* the rejection-sampling step itself — you can use a frozen, strongly-prompted model (IRCoT-style) as the chain-sampling policy during CoRAG's training-data construction phase, since it already produces reasonable interleaved retrieval-and-reasoning traces. Only chains that reach the correct final answer are then kept and used to fine-tune a smaller/cheaper target model, which — once trained — can be deployed without needing the elaborate IRCoT prompt scaffold at all. This lets you pay the prompting/frozen-model inference cost once, during offline data generation, in exchange for a cheaper, faster, natively-reformulating model in production.

</details>

---

## Q6. Walk through the CoRAG architecture end-to-end, from training to inference. `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

```
Training-time: building retrieval-chain data via rejection sampling
Existing RAG dataset (query -> final answer only)
  → sample MANY candidate retrieve-then-reason chains per question
  → reject chains that don't reach the correct answer, keep ones that do
  → fine-tune the model on (query, kept chain, final answer) triples

Inference-time: test-time compute scaling
Query → Retrieval Chain (learned, not prompted):
  step 1: reformulate query given state → retrieve → sub-answer
  step 2: reformulate query given updated state → retrieve → sub-answer
  step 3: ... (chain length controllable at test time)
  → Decoding strategy controls compute (greedy / best-of-N / adaptive length)
  → Final Answer Generator synthesizes from the full chain
```

The training and inference phases solve two different problems: training's rejection sampling is entirely about manufacturing a supervised signal for a skill (query reformulation) that no dataset labels directly (Q2); inference's decoding-strategy choice is entirely about trading compute for accuracy on a model that has already internalized that skill (Q3). This two-phase separation is what lets the same trained model serve multiple cost/accuracy operating points without retraining — a property CoRAG shares with reasoning models' "thinking budget" controls, applied specifically to retrieval chains rather than free-form reasoning tokens.

</details>

---

## Q7. What is the single distinctive mechanism that separates CoRAG from prompted iterative retrieval architectures? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The distinctive mechanism is **learning query reformulation from rejection-sampled chains, making chain length an explicit, trained-in test-time compute knob** — rather than relying on a frozen model's prompted, ad hoc reformulation behavior (IRCoT, file 19), CoRAG fine-tunes the model specifically on chains proven (by rejection sampling) to reach correct answers, so the reformulation skill itself is optimized against outcome correctness rather than whatever a general-purpose model happens to produce when prompted to "think step by step and search."

This has a direct, measurable consequence in Q3's decoding strategies: because the reformulation policy is learned and consistent (not dependent on how well a specific prompt happens to work with a specific frozen model), chain length and sampling strategy become calibrated, reliable levers for trading compute against accuracy — analogous to a reasoning model's inference-time compute scaling, but applied to retrieval chains specifically. A prompted system's "just add more retrieval rounds" doesn't have this same reliability, since a frozen model's reformulation quality over many rounds was never specifically optimized to remain good that far into a chain.

</details>

---

## Q8. How does CoRAG compare to Search-R1 (#42)? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Both train a model to make better retrieval decisions rather than relying on prompting, but use different training signals to get there. Search-R1 (#42) uses reinforcement learning (PPO/GRPO) against a sparse outcome reward, learning through many rollouts and policy-gradient updates whether a given search decision contributed to a correct final answer. CoRAG uses rejection sampling plus standard supervised fine-tuning — sample many candidate chains, keep only the ones that already reached the correct answer, and imitate those via ordinary SFT, with no policy-gradient RL step at all.

This makes CoRAG's training pipeline considerably simpler and cheaper to implement than Search-R1's full RL infrastructure (no reward model, no PPO/GRPO trainer, no rollout-based policy gradient), at the cost of being less able to learn from *partial* credit — rejection sampling is binary (a chain is kept or discarded based on final-answer correctness alone), whereas RL's reward signal can in principle shape behavior more gradually. Both share the same rejection-sampling-flavored bootstrapping problem: neither has ground-truth intermediate labels, and both solve it by deriving a usable training signal purely from final-answer correctness (Q2, and Search-R1's outcome reward, #42 Q4).

</details>

---

## Q9. What is the research origin of CoRAG, and what results does the paper report? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

CoRAG was introduced by Wang, Chen, Yang, Huang, Dou & Wei, *Chain-of-Retrieval Augmented Generation* (Microsoft Research, arXiv:2501.14342, January 2025; accepted at NeurIPS 2025), trained and evaluated on KILT (Knowledge Intensive Language Tasks) benchmark tasks, augmented with retrieval chains constructed via the rejection-sampling procedure in Q2.

The paper's headline results include establishing new state-of-the-art performance across KILT's diverse knowledge-intensive tasks, with a reported greater than 10-point exact-match improvement over strong baselines specifically on multi-hop QA when using best-of-N sampled-chain decoding (Q3) — the largest gains concentrated exactly where multi-step retrieval reformulation matters most, consistent with the architecture's core premise that learned, evolving-state-conditioned reformulation outperforms either single-shot retrieval or prompted, unoptimized multi-hop reformulation.

</details>

---

## Q10. What are the key tuning knobs for CoRAG, and how do you choose them? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Knob | Effect | Starting point |
|---|---|---|
| `n_samples` (candidate chains per training question, Q2) | More samples increase the odds of finding at least one correct chain per question, improving training data coverage, but multiply data-generation cost | 20, per Q2's pseudocode; raise for harder questions with a lower per-sample success rate |
| `max_hops` (chain length during sampling and inference) | Longer chains handle deeper multi-hop questions but increase drift risk (Q4) and cost | 4-6, tuned to your domain's genuine multi-hop depth |
| Decoding strategy (greedy / best-of-N / adaptive, Q3) | The primary inference-time cost/accuracy dial | Adaptive length as the default; best-of-N reserved for a query segment identified as needing maximum accuracy |
| Sampling temperature (Q2's `temperature=0.8` during training-data generation) | Higher temperature produces more diverse candidate chains, improving the odds rejection sampling finds varied successful strategies rather than one narrow pattern | 0.7-0.9 is typical for diversity-seeking sampling; too low risks generating too few distinct chains to learn a robust reformulation policy from |

`n_samples` directly trades training-data-generation cost against training-data coverage and quality — for questions where the correct answer is hard to reach via any chain (very difficult multi-hop questions), a low `n_samples` may yield zero accepted chains for that question, silently excluding the hardest examples from the training set entirely, which is worth monitoring (Q12) rather than assuming rejection sampling always succeeds given enough attempts.

</details>

---

## Q11. How do you evaluate a CoRAG-trained model's performance? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The standard evaluation follows the paper's own methodology (Q9): exact-match and related metrics on KILT's knowledge-intensive tasks, with results segmented by decoding strategy (greedy vs. best-of-N vs. adaptive, Q3) to characterize the actual compute/accuracy trade-off curve for your specific fine-tuned model rather than assuming the paper's reported curve transfers unchanged. For a domain-specific deployment (Q18), build a golden set from your own multi-hop question distribution, since KILT's task mix may not represent your domain's actual reformulation patterns.

Beyond final-answer accuracy, track **chain quality metrics** specifically: average chain length actually used (for adaptive decoding, does the model correctly use fewer hops on easy questions and more on hard ones, or does it use a roughly constant length regardless of difficulty); and **drift rate** (Q4) — the fraction of chains where an early sub-answer error propagates into later reformulations, measurable by comparing intermediate sub-answers against ground truth where available. A model with strong final-answer accuracy but a high drift rate that happens to self-correct late in the chain is a fundamentally less robust policy than one that stays on-track throughout, even if both currently score similarly on the headline metric.

</details>

---

## Q12. What is the characteristic failure mode of rejection sampling accepting a "right answer, wrong reasoning" chain, and how do you detect it? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Rejection sampling's filter is binary and outcome-only (Q2): a chain is kept if its final answer matches the gold label, with no verification that the intermediate reformulations and sub-answers were actually correct or sensible along the way. This means a chain that reaches the right final answer partly by chance — a garbled or off-topic reformulation at hop 2 that, coincidentally, still retrieves a passage useful enough for the model to recover and land on the correct final answer anyway — gets accepted into the training set exactly as if every step had been reasoned soundly, teaching the fine-tuned model to imitate a pattern that only worked by luck in that specific instance.

**Symptom:** the fine-tuned model can learn a reformulation policy that "gets away with" occasional low-quality intermediate steps because rejection sampling's training signal never distinguished a lucky recovery from a genuinely well-reasoned chain — this can manifest as the model's chain-level quality (Q11's drift-rate tracking) being worse than its final-answer accuracy alone would suggest, since final-answer accuracy averages over cases where the model got lucky and cases where it reasoned soundly. **Detection:** for training chains, before or after filtering by final-answer match, additionally sample a check on intermediate sub-answer quality (an LLM-judge pass scoring each hop's reformulation and sub-answer for topical relevance to the original question) — a chain that passes the final-answer filter but scores poorly on intermediate-step quality is a candidate for exclusion or down-weighting, tightening the training signal beyond outcome-only filtering.

</details>

---

## Q13. How do you implement the reranking step in best-of-N decoding? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

`model.rerank_and_select` (Q3's `corag_best_of_n`) needs a way to score N complete candidate chains and pick the best one, without access to ground truth at inference time (unlike training's rejection sampling, which has gold answers to filter against). A practical implementation scores each candidate chain along dimensions correlated with quality rather than correctness directly:

```python
def rerank_and_select(candidates: list[dict], model) -> str:
    """Score N candidate chains and select the best without ground truth."""
    scored = []
    for candidate in candidates:
        # Self-consistency: does the model's own confidence in the final
        # answer, given the full chain, look high?
        confidence = model.score_answer_confidence(candidate["final_answer"], candidate["chain"])

        # Chain coherence: do the sub-answers build on each other sensibly,
        # or does the chain show signs of drift (Q4)?
        coherence = model.score_chain_coherence(candidate["chain"])

        # Retrieval support: are the retrieved passages at each hop
        # genuinely relevant to that hop's reformulated query?
        retrieval_quality = mean(
            model.score_relevance(step["query"], step["passages"]) for step in candidate["chain"]
        )
        scored.append((candidate, confidence * 0.5 + coherence * 0.3 + retrieval_quality * 0.2))

    best_candidate, _ = max(scored, key=lambda x: x[1])
    return best_candidate["final_answer"]
```

The most common practical approach, and the one closest to the paper's own framing, is **self-consistency voting**: if multiple sampled chains independently arrive at the same final answer, that answer is more likely correct than one only one chain reached — a cheap, effective proxy that doesn't require a separately-trained reranking model, though a learned reranker (trained on chain-quality-vs-correctness data, similar in spirit to a reward model) can outperform simple voting when chains disagree with no clear majority.

</details>

---

## Q14. How does CoRAG's learned reformulation differ from RQ-RAG's (#51) special-token query-refinement operations? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

RQ-RAG (#51) fine-tunes a model to explicitly choose among a small, named set of query-refinement operations (rewrite, decompose, disambiguate) via special output tokens — the reformulation strategy is constrained to an interpretable, fixed vocabulary of operations, and the training signal teaches *which operation* to apply when. CoRAG's reformulation is unconstrained free-form text generation, learned purely from which complete chains happened to reach the correct answer via rejection sampling (Q2) — there's no explicit operation vocabulary at all, just an implicit, emergent reformulation policy shaped by outcome-filtered imitation.

The trade-off mirrors the general interpretability-vs-flexibility pattern seen elsewhere in this bank (e.g., DeepRAG's #49 structured subqueries vs. Auto-RAG's free-form dialogue, #49 Q13): RQ-RAG's special-token operations are individually inspectable (you can see which operation the model chose and audit whether it made sense), while CoRAG's reformulations are flexible and can adapt to patterns no fixed operation vocabulary anticipated, at the cost of being harder to audit or constrain to a known-safe set of behaviors.

</details>

---

## Q15. How would you build a decision-gate benchmark to decide whether CoRAG's rejection-sampling training investment is worth it over IRCoT-style prompting? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Q5 already lays out the qualitative conditions favoring each approach; a decision gate makes the comparison quantitative for your specific deployment:

```
1. Confirm preconditions: do you have (or can you generate, per Q2) a
   sufficiently large RAG dataset with final-answer labels, and access
   to fine-tune a base model? If either is missing, this gate fails
   immediately and IRCoT-style prompting is the only viable option.

2. Baseline: measure a well-prompted IRCoT-style system's accuracy and
   per-query latency/cost on your actual multi-hop query distribution.

3. Bootstrap training data (Q2, Q5's combination idea): use the IRCoT
   baseline itself as the chain-sampling policy for rejection sampling,
   reusing the baseline's own inference infrastructure rather than
   building a separate sampling pipeline from scratch.

4. Fine-tune a candidate CoRAG-style model on the resulting chains, and
   measure the same accuracy/latency/cost metrics on the same eval set.

5. Compare: at your expected query volume, does the fine-tuned model's
   lower per-query cost (Q16, no elaborate prompt scaffold needed at
   inference) amortize the one-time rejection-sampling-and-training
   cost within an acceptable payback period, while matching or
   exceeding the IRCoT baseline's accuracy?

6. Gate: proceed to production CoRAG deployment only if the fine-tuned
   model's accuracy holds up (per Q11's chain-quality metrics, not just
   final-answer accuracy) AND the cost/latency advantage at your volume
   justifies the training investment and the retraining-cycle cost of
   maintaining it going forward.
```

This mirrors the same data-availability-first gating discipline used for Search-R1's training-investment decision (#42 Q15), with the added wrinkle that Q5's bootstrapping idea (use IRCoT to generate CoRAG's training data) lowers the barrier to attempting step 3-4 even for teams without an existing gold-chain dataset.

</details>

---

## Q16. What is the cost and infrastructure overhead of rejection-sampling data construction and training at scale? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Rejection sampling's cost scales with `n_training_questions x n_samples` (Q10) — each candidate chain requires `max_hops` reformulation calls, retrieval calls, and sub-answer generation calls, meaning data construction alone can require a very large number of LLM and retrieval calls before any fine-tuning even begins. Illustrative cost at 10,000 training questions, `n_samples=20`, `max_hops=4`: roughly 800,000 reformulation/sub-answer generation calls plus 800,000 retrieval calls just for data construction — a substantial one-time cost, though smaller than Search-R1's comparable RL rollout cost (#42 Q16) since CoRAG's rejection sampling doesn't require GRPO's multiple-rollouts-per-prompt-for-relative-advantage structure, only enough samples per question to find at least one accepted chain.

Beyond raw compute, infrastructure needs include: a sampling pipeline capable of running many chains in parallel against a live (or snapshot) retriever during data construction; storage for the resulting chain dataset; and a standard supervised fine-tuning pipeline (simpler than Search-R1's RL training infrastructure, Q8) for the actual model training step. At inference/serving time, cost is comparable to or cheaper than a prompted alternative for the same chain length, since a fine-tuned model doesn't carry an elaborate prompt scaffold's token overhead on every call — the cost asymmetry strongly favors CoRAG at high query volume (Q15), where the one-time data-construction and training cost amortizes, and disfavors it at low volume, where the upfront investment may never pay back.

</details>

---

## Q17. What security and trust risks are specific to rejection-sampled training chains? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

- **Spurious-correct chains teaching bad habits at scale** — Q12's "right answer, wrong reasoning" risk is a quality issue in isolated cases, but if it occurs systematically for a particular question type or corpus region, the fine-tuned model can learn a genuinely unreliable reformulation pattern that happens to correlate with correct answers in the training distribution without being a sound strategy in general — a subtler training-data-quality risk than obvious mislabeling, since each individual accepted chain did technically satisfy the acceptance criterion.
- **Poisoned training-time retrieval corpus** — since chain sampling (Q2) retrieves against a real corpus during data construction, a poisoned document encountered during training-data generation could get incorporated into an accepted chain (if the chain still happens to reach the correct final answer despite the poisoned content, or worse, because of a coincidental interaction with it), teaching the fine-tuned model a reformulation habit shaped in part by content that shouldn't have been trusted — the same training-time corpus poisoning concern raised for Search-R1 (#42 Q17), applicable here via the sampling process instead of an RL rollout.
- **Gold-answer quality directly gates training-data quality** — since acceptance is purely outcome-based (Q2's `answers_match`), any errors or ambiguity in the gold-answer labels used for rejection sampling propagate directly into which chains get accepted or rejected, with no independent check — a mislabeled gold answer could cause a genuinely well-reasoned chain to be incorrectly rejected (lost training signal) or a poorly-reasoned chain that happens to match the wrong gold label to be incorrectly accepted (bad training signal), mirroring the gold-label-noise risk described for Search-R1 (#42 Q19).
- **Reranker gaming at inference time** (Q13) — if best-of-N reranking relies on a learned or heuristic scoring function, a chain that produces text patterns the reranker over-weights (confident-sounding phrasing, superficial coherence markers) without being genuinely more likely correct could be selected over a better but less "confident-sounding" chain, an inference-time analog of reward hacking.

Mitigation: validate gold-answer quality in the source dataset before treating it as ground truth for rejection sampling; screen the training-time retrieval corpus for the same poisoning risks as any production corpus; and supplement outcome-only chain acceptance with the intermediate-quality spot-checking described in Q12, rather than trusting final-answer match as a fully sufficient training-data quality gate.

</details>

---

## Q18. Design a domain-adapted CoRAG system for a legal or medical multi-hop research assistant. `[Advanced]` `[Scenario]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Requirements:** multi-hop questions requiring chained retrieval across case law or clinical literature ("what precedent, established by the case cited in this ruling, applies to the current fact pattern"); domain-specific reformulation idioms (legal citation chains, clinical terminology) that a general-purpose model's prompted reformulation wouldn't naturally produce; high enough query volume to justify fine-tuning investment (Q15).

```
1. Training data construction (Q2): sample candidate chains against your
   OWN domain corpus (case law database, clinical literature), not a
   general corpus like KILT/Wikipedia -- domain-specific reformulation
   idioms can only be learned from domain-specific rejection-sampled
   chains, per Q5's "domain-specific query reformulation patterns"
   condition for choosing CoRAG over prompting.

2. Gold-answer quality control (Q17): given the stakes, have domain
   experts verify a sample of gold answers used for rejection sampling
   before trusting the resulting training data at scale -- an
   incorrect gold label in this domain risks training a systematically
   wrong reformulation habit into the model.

3. Intermediate-quality auditing (Q12): given the cost of a spurious-
   correct chain in a high-stakes domain, apply the LLM-judge
   intermediate-step quality check as a mandatory filter, not an
   optional enhancement, before accepting a chain into training data.

4. Decoding strategy (Q3, Q10): default to adaptive-length decoding for
   routine queries, but route queries flagged as high-stakes (per
   Verifiable RAG's #33 domain precedent) to best-of-N with reranking
   (Q13), accepting the higher cost for the accuracy ceiling it provides
   on consequential questions.

5. Chain-quality monitoring in production (Q11): track drift rate and
   average chain length by query category on an ongoing basis, since a
   domain-adapted model's reformulation quality may degrade on novel
   query patterns not well-represented in the original training
   chains, requiring periodic re-training with fresh rejection-sampled
   data as the domain corpus and typical queries evolve.
```

The key design choice is domain-specific training-data construction combined with expert-verified gold answers — CoRAG's entire value proposition depends on the quality of what rejection sampling accepts, and in a high-stakes domain, the ordinary "outcome-match is good enough" filtering bar (Q12, Q17) needs to be raised with additional verification specifically because the cost of a subtly-wrong learned reformulation habit is much higher here than in a general-purpose QA setting.

</details>

---

## Q19. What happens when best-of-N reranking itself is unreliable, and how do you debug it? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

If `rerank_and_select` (Q13) systematically picks a worse chain over a better one — because its scoring signals (confidence, coherence, retrieval quality) don't actually correlate well with true correctness for your domain — best-of-N decoding can underperform even greedy single-chain decoding despite spending N times the compute, since sampling more candidates only helps if the selection step can reliably identify the best one among them. This is a distinct failure from chain drift (Q4): the chains themselves may include a genuinely correct one, but reranking fails to surface it.

**Debugging:** (1) on a labeled evaluation set, measure reranking accuracy directly — among N sampled chains for a question with known ground truth, does the reranker's top pick match the chain that actually reached the correct answer, tracked as its own metric separate from overall best-of-N pipeline accuracy; (2) if reranking accuracy is poor, test simple self-consistency voting (majority final-answer agreement across chains, Q13) as a baseline — if voting alone outperforms the more elaborate scoring-function reranker, the additional signals (coherence, retrieval quality) may be adding noise rather than useful signal for your specific model and domain; (3) if a learned reranker is in use, audit it the same way any reward-model-like component would be audited (Q19 of Search-R1, #42) — check for systematic biases toward superficial confidence or fluency markers rather than genuine correctness indicators, and retrain or recalibrate against a corrected training signal if bias is confirmed.

</details>

---

## Q20. What are the limitations of CoRAG, and how might the field evolve? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Current limitations: (1) **rejection sampling's outcome-only filter can accept spuriously-correct chains** (Q12, Q17) — with no verification of intermediate-step quality unless explicitly added, teaching the model habits that only worked by chance in specific training instances; (2) **chain drift remains a real risk even after mitigation** (Q4) — rejection sampling and best-of-N reduce but don't eliminate the fundamental risk that an early reformulation error compounds through a chain; (3) **best-of-N's accuracy ceiling depends entirely on reranking quality** (Q13, Q19) — sampling more chains without a reliable way to select the best one wastes the additional compute; (4) **data construction cost is substantial and domain-specific** (Q16, Q18) — a general-purpose training run doesn't transfer domain-specific reformulation idioms, requiring fresh rejection sampling for each new domain.

Likely evolution: **process-level supervision** replacing or supplementing today's outcome-only rejection filtering — verifying intermediate reformulation and sub-answer quality during data construction (Q12's mitigation, made standard rather than optional) to produce cleaner training signal; **learned rerankers trained specifically for chain selection** (rather than heuristic scoring functions, Q13) as best-of-N decoding matures into a more standard production pattern; and continued convergence with the RL-trained search family (Search-R1, #42) — since both CoRAG and Search-R1 solve the same underlying "no ground-truth intermediate labels" problem with different tools (rejection sampling plus SFT vs. RL against outcome reward), a natural evolution is hybrid training pipelines that use rejection sampling to bootstrap an initial policy cheaply, then refine it further with RL for the harder residual cases rejection sampling alone doesn't adequately cover.

</details>

---

## Real-World Applications

- **Enterprise multi-hop search assistants at scale**: fine-tuned CoRAG-style models reduce per-query latency/cost versus prompting-based interleaved retrieval (IRCoT) when query volume is high
- **Knowledge-intensive benchmarks (KILT tasks)**: CoRAG established new state-of-the-art results across a diverse set of knowledge-intensive tasks by training directly on rejection-sampled chains
- **Cost-tiered QA products**: exposing chain-length/sampling as a user- or product-tier-selectable knob (fast/cheap vs. thorough/expensive answers), the same way reasoning-effort tiers work for reasoning models
- **Domain-adapted retrieval assistants** (legal, medical, financial multi-hop research): fine-tuning the reformulation policy on domain-specific rejection-sampled chains to learn domain query-reformulation idioms a frozen general-purpose model wouldn't produce via prompting alone
