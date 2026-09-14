# 42 — Search-R1 / Reasoning RAG (RL-Trained Search)

> An LLM is trained end-to-end with reinforcement learning to interleave its own reasoning tokens with self-issued search calls, learning *when* and *what* to retrieve purely from answer-correctness reward — no one hand-writes the retrieval policy.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
Query
    │
    ▼
Policy LLM generates ──► <think> reasoning tokens </think>
    │
    ▼
Policy LLM decides to retrieve ──► <search> query </search>
    │
    ▼
Retriever executes the query, returns passages ──► <information> ... </information>
    │           (inserted into context, NOT produced by the LLM)
    ▼
Policy LLM continues ──► more <think>/<search> turns, as many as needed
    │
    ▼
Policy LLM emits <answer> final answer </answer>
    │
    ▼
Outcome Reward (exact-match / F1 against gold answer) ──► PPO / GRPO update
    (retrieved-token loss masking: gradient only flows through the
     LLM's own think/search/answer tokens, never the inserted passages)
```

### Key Components

| Component | Responsibility |
|---|---|
| Policy LLM | Single model that both reasons and decides when/what to search, trained end-to-end |
| Retriever (frozen) | Dense/sparse search engine invoked as an external tool; not trained, just called |
| Rollout Engine | Executes multi-turn think→search→observe trajectories during RL training and inference |
| Reward Function | Outcome-based (answer correctness, e.g. F1/EM); no process supervision on retrieval decisions |
| RL Trainer (PPO/GRPO) | Updates policy weights using rollout reward, with retrieved-token loss masking |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| RL algorithm | PPO, GRPO (Group Relative Policy Optimization) |
| Reference implementation | Search-R1 (PeterGriffinJin/Search-R1, GitHub) |
| RL training infra | veRL, OpenRLHF, TRL |
| Retriever backend | Dense retriever (e.g. E5) over Wikipedia dumps, or a live search API |
| Sibling frameworks | R1-Searcher, ReSearch, DeepRetrieval (same RL-for-search family) |

---

## Q1. What is Search-R1 and how does it differ from prompted Agentic RAG? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Search-R1** (Jin et al., 2025, *"Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning"*, arXiv:2503.09516) trains an LLM with reinforcement learning to autonomously interleave reasoning (`<think>`) and search calls (`<search>`) during generation, optimizing purely for final-answer correctness. No human ever writes rules for *when* to search — the policy learns it from reward.

This is a fundamentally different mechanism from the prompted **Agentic RAG** described in file 04, which uses **ReAct-style or FLARE-style prompting**: a frozen, instruction-following LLM is *told* (via prompt engineering) to reason, then act, then observe, in a fixed loop. The retrieval policy in Agentic RAG lives entirely in the prompt; the model's weights never change.

```
Agentic RAG (file 04) — PROMPTED policy:
  System prompt: "Think step by step. If you need information, call search(query).
                   Otherwise, answer directly."
  Frozen LLM (e.g. Claude, GPT-4) follows this instruction each turn.
  Retrieval decisions = whatever the frozen model infers from the prompt.

Search-R1 — LEARNED policy:
  Base LLM (e.g. Qwen2.5-7B) is fine-tuned with RL.
  Reward = 1 if final answer matches gold, else 0.
  Over thousands of rollouts, gradient updates shape WHEN the model emits
  <search> vs continues reasoning — this becomes part of the model's weights,
  not a prompt instruction.
```

| Dimension | Agentic RAG (file 04, ReAct/FLARE) | Search-R1 (RL-trained) |
|---|---|---|
| Retrieval policy | Prompted (in-context instructions) | Learned (baked into model weights via RL) |
| Requires training run | No | Yes (PPO/GRPO over rollouts) |
| Generalizes retrieval timing | Only as well as the prompt generalizes | Learned from reward signal across many examples |
| Model needed | Any instruction-following LLM | A model you can fine-tune (open weights) |
| Reported gains | Depends on prompt engineering | Search-R1 reports +41% (Qwen2.5-7B) / +20% (Qwen2.5-3B) over RAG baselines on 7 QA benchmarks |

**Key insight:** Agentic RAG asks a frozen model to *follow* a retrieval strategy; Search-R1 makes the model *discover* its own retrieval strategy through trial and error, guided only by whether the final answer was correct.

</details>

---

## Q2. What does the training rollout actually look like, token by token? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A Search-R1 rollout is a single generation stream where the policy LLM's own tokens are interleaved with retriever output that gets *inserted*, not generated:

```
<think> The question asks about a treaty signed in 1919. I should search for it. </think>
<search> Treaty of Versailles signatories 1919 </search>
<information> [Retriever executes the query against the index]
  Doc 1: "The Treaty of Versailles was signed on 28 June 1919 by..."
  Doc 2: "Signatories included the Allied Powers and Germany..."
</information>
<think> The passages confirm Germany and the Allied Powers signed it.
  I now have enough to answer. </think>
<answer> The Treaty of Versailles (1919) was signed by the Allied Powers
  and Germany. </answer>
```

**Multi-turn behavior:** the model can emit multiple `<search>` blocks in sequence if the first retrieval was insufficient — this is what lets Search-R1 handle multi-hop questions (compare with the fixed decompose-then-retrieve pattern of file 19, Iterative Multi-Hop RAG) without any hand-coded stopping rule; the model itself learns when it has "enough."

**Pseudo-code for the rollout loop:**

```python
def rollout(question: str, policy_llm, retriever, max_turns: int = 4) -> str:
    context = f"Question: {question}\n"
    for turn in range(max_turns):
        # Policy generates until it hits </search>, </answer>, or max tokens
        segment = policy_llm.generate(context, stop=["</search>", "</answer>"])
        context += segment

        if segment.strip().endswith("</search>"):
            query = extract_between(segment, "<search>", "</search>")
            passages = retriever.search(query, k=3)
            info_block = f"<information>{format_passages(passages)}</information>\n"
            context += info_block          # inserted, not generated by the model
        elif segment.strip().endswith("</answer>"):
            return extract_between(segment, "<answer>", "</answer>")
    return "NO_ANSWER"
```

**Reward is computed only at the end**, comparing the extracted `<answer>` against the gold label (exact match or F1) — there is no intermediate reward for "good" search queries.

</details>

---

## Q3. Why does Search-R1 mask the retrieved-token loss during training, and what breaks if you don't? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

During RL fine-tuning, the policy gradient is computed over the *log-probability of the tokens the model generated*. The `<information>...</information>` block is not generated by the model — it's copied verbatim from the retriever's output and spliced into the context. If you don't mask it, the training loop will treat those retrieved tokens as if the model "chose" to produce them, and will:

```
Without masking:
  - The policy gradient tries to increase/decrease the log-probability of
    retrieved passage tokens (e.g. "The Treaty of Versailles was signed on...")
  - But the model never actually predicted those tokens — they're copy-pasted
  - This injects noisy, meaningless gradient signal tied to whatever documents
    happened to be retrieved, unrelated to the model's own decisions
  - Training becomes unstable; loss spikes correlate with long/short retrieved
    passages rather than actual policy quality

With retrieved-token loss masking:
  - Loss mask = 0 for all <information>...</information> tokens
  - Loss mask = 1 for <think>, <search>, <answer> tokens (the model's own output)
  - Gradient only updates the model's reasoning/search/answer behavior
  - Training signal is clean: "did MY decisions lead to a correct answer?"
```

```python
def compute_loss_mask(token_ids: list[int], info_start_id: int, info_end_id: int) -> list[int]:
    """1 = model-generated token (train on it), 0 = retrieved/inserted token (mask it)."""
    mask, inside_info = [], False
    for tok in token_ids:
        if tok == info_start_id:
            inside_info = True
        mask.append(0 if inside_info else 1)
        if tok == info_end_id:
            inside_info = False
    return mask

# loss = -sum(logprob[i] * mask[i] for i in range(len(tokens))) / sum(mask)
```

This is the same principle as masking prompt tokens in standard SFT — you only backpropagate through what the model is responsible for producing.

</details>

---

## Q4. What outcome-based reward function does Search-R1 use, and why not reward the search queries directly? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Search-R1 uses a deliberately simple **outcome reward**: compare the final extracted `<answer>` to the gold answer using exact match (EM) or F1, with a small format-compliance bonus/penalty for well-formed `<think>/<search>/<answer>` tags.

```python
def compute_reward(rollout_output: str, gold_answer: str) -> float:
    format_ok = has_valid_tags(rollout_output)          # <think>, <search>, <answer> well-formed
    if not format_ok:
        return -1.0                                       # format penalty
    predicted = extract_between(rollout_output, "<answer>", "</answer>")
    em = int(normalize(predicted) == normalize(gold_answer))
    return float(em)                                       # 0 or 1
```

**Why not reward the search queries directly (e.g. reward retrieval precision/recall)?**

- Building a "good query" labelset requires humans to annotate what an ideal search query looks like for every training question — expensive and subjective.
- Rewarding retrieval metrics directly can be gamed: the model learns to produce queries that maximize BM25/embedding overlap with a labeled "good" passage without that passage actually being useful for answering.
- Outcome-based reward is **self-supervising**: you only need (question, gold answer) pairs, which already exist in QA datasets — no retrieval-quality annotation needed.
- It naturally handles the multi-hop case: an intermediate search doesn't need to be individually "correct," it just needs to contribute to an eventually correct final answer, so the model is free to discover unconventional but effective query strategies.

**Trade-off:** with sparse, delayed reward (only 0/1 at the very end of a multi-turn trajectory), credit assignment is harder — GRPO variants (used by Search-R1 and R1-Searcher) address this by comparing multiple rollouts of the same question against each other (relative advantage) rather than relying on a learned value function like standard PPO.

</details>

---

## Q5. What are the failure modes of RL-trained search policies, and how do they compare across the Search-R1 family? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Failure mode 1 — reward hacking on retrieval count.** With a pure outcome reward, a model can learn to over-search (issue many redundant queries "just in case," inflating latency/cost) or under-search (guess from parametric memory when a lucky guess is rewarded, skipping retrieval on questions that actually needed it). Mitigations: add small per-search-call cost penalties, or format/turn-count regularization.

**Failure mode 2 — training instability over long multi-turn trajectories.** As the number of `<search>` turns grows, the effective trajectory length balloons, and sparse terminal reward makes credit assignment noisy. GRPO-style methods stabilize this by normalizing reward within a group of rollouts for the same prompt, rather than requiring a separate learned critic (as vanilla PPO does).

**Failure mode 3 — reliance on a frozen retriever's blind spots.** Because the retriever itself is not trained, the policy can only learn to phrase *queries* better — it cannot fix a fundamentally weak or stale index. If the retriever consistently fails on a class of questions, the RL policy will learn to route around it (answer from parametric knowledge) rather than fixing the underlying retrieval gap, which can silently increase hallucination on that class of questions.

**How the family compares:**

| Method | Distinctive mechanism | Reward signal |
|---|---|---|
| **Search-R1** | Multi-turn `<think>/<search>/<answer>`, retrieved-token masking | Outcome (EM/F1) + format |
| **R1-Searcher** | Two-stage RL: first learns to invoke search reliably, then optimizes answer quality | Stage 1: search-format reward; Stage 2: outcome reward |
| **ReSearch** | Frames search as a first-class reasoning-chain operation, elicits emergent reflection/self-correction without heuristics | Outcome-only, no supervised reasoning traces |
| **DeepRetrieval** | Trains the *query generator* itself (not just an answering agent) against real search engine APIs, rewarded by retrieval metrics (recall/relevance) | Retrieval-quality reward (recall@k), not final-answer reward |

**Combining with other bank architectures:** an RL-trained search policy like Search-R1 can be layered underneath a Corrective RAG (file 06) verification step — the learned policy decides *when* to retrieve, while a separate lightweight judge still checks *whether* the retrieved evidence is sufficient before allowing an `<answer>`, combining a learned retrieval trigger with an explicit correction loop.

</details>

---

## Q6. Walk through the Search-R1 architecture end-to-end. `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

```
Query
    │
    ▼
Policy LLM generates ──► <think> reasoning tokens </think>
    │
    ▼
Policy LLM decides to retrieve ──► <search> query </search>
    │
    ▼
Retriever executes the query, returns passages ──► <information> ... </information>
    │           (inserted into context, NOT produced by the LLM)
    ▼
Policy LLM continues ──► more <think>/<search> turns, as many as needed
    │
    ▼
Policy LLM emits <answer> final answer </answer>
    │
    ▼
Outcome Reward (exact-match / F1 against gold answer) ──► PPO / GRPO update
```

The architecture has exactly one trained component (the policy LLM) and one frozen component (the retriever, Q1) — everything about *when* and *what* to search lives in the policy's weights, learned end-to-end from the single outcome signal at the very end of the trajectory. This is structurally the simplest architecture in this bank's agentic/reasoning family in terms of moving parts, precisely because the complexity that a prompted system pushes into careful prompt engineering (file 04) is instead pushed into the training process itself.

</details>

---

## Q7. How does Search-R1 compare to WebGPT (#39), another RL-trained retrieval architecture? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Both train a policy end-to-end with RL to make retrieval decisions, but differ in action space and reward structure. WebGPT (#39) learns a **fixed, structured browsing action space** (search/click/scroll/quote/done) trained via behavior cloning on human demonstrations, then RLHF against human preference comparisons between full transcripts — the reward signal is human judgment of overall answer quality. Search-R1 uses a **simpler, unstructured action space** (just `<think>` and `<search>`, no click/scroll/quote), trained purely against automatic outcome correctness (exact-match/F1 against a gold answer, Q4) — no human preference labeling at all.

This difference in reward source is the more consequential one: WebGPT's human-preference reward captures nuanced qualities (citation quality, tone) that automatic correctness metrics miss, but requires expensive human labeling; Search-R1's automatic EM/F1 reward is essentially free to compute at scale (any QA dataset with gold answers works) but only optimizes for final-answer correctness, with no direct signal about *how* the model got there, browsing quality, or citation reliability along the way.

</details>

---

## Q8. What is the research origin of Search-R1, and what result does the paper report? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Search-R1 was introduced by Jin et al., *Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning* (arXiv:2503.09516, 2025), building on the reasoning-via-RL recipe popularized by DeepSeek-R1 (which trained models to produce long `<think>` chains purely from outcome reward, without supervised reasoning traces) and extending it specifically to interleave search calls within that same reasoning process.

The paper's headline result: RL-trained interleaved reasoning-and-search reported gains of roughly +41% (Qwen2.5-7B) and +20% (Qwen2.5-3B) over prompted RAG baselines, averaged across seven QA benchmarks including multi-hop datasets like HotpotQA, 2WikiMultihopQA, and Musique — notable both for the size of the improvement and for demonstrating that a relatively small (3B-7B parameter) open-weight model, trained this way, could substantially outperform simply prompting the same or a larger model to follow a retrieve-then-generate pattern.

</details>

---

## Q9. How does Search-R1 compare to RQ-RAG (#51), which also fine-tunes a model for retrieval decisions? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Both fine-tune a model to make better retrieval decisions rather than relying on prompting, but target different sub-problems. RQ-RAG (#51) fine-tunes a model to explicitly chain **query-refinement operations** (rewrite, decompose, disambiguate) via special tokens — the model learns *how to reformulate* a query into better sub-queries, with the actual retrieve-and-answer loop otherwise structured similarly to standard iterative RAG. Search-R1 fine-tunes a model to decide **when to search at all** within a free-form reasoning trace, interleaving arbitrary `<think>` and `<search>` calls with no constraint on query reformulation strategy specifically — the model can phrase queries however its RL training found effective, with no special-token vocabulary dedicated to reformulation operations.

The distinction mirrors the reward-signal difference in Q7: RQ-RAG's special-token approach makes the query-refinement strategy interpretable and constrained (you can inspect which operation the model chose), while Search-R1's approach is more flexible but less interpretable (the model's query phrasing strategy is whatever emerged from outcome-reward optimization, with no explicit vocabulary describing *why* it phrased a query a particular way).

</details>

---

## Q10. What are the key tuning knobs for a Search-R1 training run, and how do you choose them? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Knob | Effect | Starting point |
|---|---|---|
| `max_turns` (Q2's rollout loop) | More turns allow deeper multi-hop reasoning but increase rollout cost and credit-assignment difficulty (Q12) | 4, per Q2's pseudocode; raise for domains with genuinely deep multi-hop needs |
| `k` (passages per search call) | More passages give richer context per search but increase trajectory length and per-turn token cost | 3, per Q2's default |
| Format-compliance penalty weight (Q4) | Determines how strongly malformed `<think>/<search>/<answer>` tags are penalized relative to answer correctness | Set high enough that format violations are reliably eliminated early in training, without dominating the correctness signal once format compliance is learned |
| GRPO group size (rollouts per prompt) | More rollouts per prompt give a better relative-advantage estimate but multiply training compute | Typically 4-16 rollouts per prompt in GRPO-style training, balancing gradient quality against compute budget |

`max_turns` and GRPO group size interact with training cost multiplicatively (longer trajectories x more rollouts per prompt), which is the primary lever for controlling the overall training compute budget — teams typically start with a smaller group size and fewer max turns to validate the training pipeline works at all before scaling either up for a full training run.

</details>

---

## Q11. How do you evaluate a Search-R1-trained model's performance? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The standard evaluation follows exactly the reward function's own metric (Q4) applied to held-out data: exact-match and F1 against gold answers on QA benchmarks, most commonly multi-hop datasets (HotpotQA, 2WikiMultihopQA, Musique) where the reported headline gains (Q8) were measured, since single-hop factual QA doesn't exercise the multi-turn search behavior Search-R1 is designed to learn.

Beyond raw accuracy, production evaluation should also track: **average search-call count per question** (a proxy for efficiency — a model achieving similar accuracy with fewer searches is cheaper to serve, and comparing this across training checkpoints reveals whether the policy is learning efficient search behavior or just brute-forcing accuracy via over-searching, Q5's failure mode 1); **format-compliance rate** (how often the model produces well-formed tags, since a format violation forfeits reward entirely per Q4); and **performance decomposed by hop count** (single-hop vs. multi-hop questions), since a model can show strong aggregate accuracy while actually only having learned the single-hop case well, with multi-hop gains concentrated in a small subset of questions.

</details>

---

## Q12. What is the characteristic failure mode of GRPO's group-relative advantage when all rollouts in a group receive the same reward? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

GRPO computes advantage by comparing a rollout's reward against the average reward of other rollouts sampled for the *same* prompt (Q4's "relative advantage" mention) — this avoids needing a separately-trained value function, but it has a specific degenerate case: if every rollout in a group receives an identical reward (e.g., a question so easy that every sampled rollout answers correctly, giving reward 1.0 across the whole group, or conversely so hard that every rollout fails, giving 0.0 across the group), the relative advantage for every rollout in that group is exactly zero — there's no learning signal at all from that group, regardless of how the individual rollouts actually differed in their reasoning or search behavior.

**Symptom:** training can plateau or show slower-than-expected progress if a large fraction of training prompts fall into this zero-variance regime — the model isn't wrong, it just isn't receiving gradient signal from those examples, which is a subtler failure than an obviously broken training run since loss curves can look superficially reasonable while a meaningful fraction of the training data isn't actually contributing useful updates. **Mitigation:** curate training data toward a difficulty distribution that reliably produces reward variance within a group (avoiding a training set dominated by trivially easy or impossibly hard questions), and monitor the fraction of zero-variance groups per training batch as a diagnostic — a rising rate over training epochs suggests the model has "solved" a growing share of the training distribution and may benefit from a harder curriculum.

</details>

---

## Q13. When would you choose Search-R1's learned policy over prompted Agentic RAG for a production system? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Choose Search-R1-style RL training when: (1) you have (or can construct) a large dataset of (question, gold answer) pairs suitable for outcome-based reward, since the entire approach depends on this being cheap and automatic (Q4); (2) you're serving high query volume where a smaller, RL-trained model's inference cost savings over repeatedly prompting a larger frontier model amortize the upfront training investment; (3) the task has genuinely learnable, generalizable search-timing patterns that a fixed prompt struggles to capture consistently (e.g., domains with unusual multi-hop structure where an off-the-shelf prompted agent's instruction-following isn't reliable).

Choose prompted Agentic RAG (file 04) when: you don't have a large gold-answer dataset (most real production tasks don't have thousands of labeled QA pairs ready-made); you need to iterate quickly on retrieval behavior (editing a prompt is same-day, retraining a policy is a multi-day-to-multi-week cycle); or the task's retrieval-timing logic is already well-captured by a capable frontier model's instruction-following, making the RL training investment's marginal benefit small. This mirrors the same learned-vs-prompted trade-off established for WebGPT vs. Agentic Web RAG (#39, #31) — the deciding factor is almost always data availability and volume economics, not raw achievable quality ceiling.

</details>

---

## Q14. How do you monitor for the frozen-retriever blind-spot failure mode in production? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Q5's failure mode 3 notes that because the retriever is never trained, the RL policy can only learn to phrase queries better against a *fixed* retrieval quality ceiling — if the retriever consistently fails on some class of questions, the policy learns to route around the gap (answering from parametric memory) rather than fixing it, which can silently increase hallucination on exactly the questions where retrieval would have mattered most.

**Monitoring:** segment production accuracy by whether the model's trajectory included a `<search>` call at all, and further by retrieval-confidence signals (did the retrieved passages score highly against the query) — a subset of questions where the model consistently *skips* search and answers directly, combined with lower accuracy on that subset relative to questions where it does search, is the signature of this failure: the policy has learned that searching for this question class doesn't help (because the retriever can't serve it well) and adapted by not bothering, which looks like efficient behavior in aggregate metrics but is actually masking a retrieval coverage gap. **Fix:** this is a retrieval-quality problem, not a policy problem (the same distinction drawn in Verifiable RAG's Q15 for unretrievable claims) — improving the underlying retriever (better embedding model, expanded index, hybrid search) is the correct lever, not further RL training of the policy against the same weak retrieval backend.

</details>

---

## Q15. How would you build a decision-gate benchmark to decide whether the RL training investment in Search-R1 is worth it? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

RL training requires infrastructure (rollout engine, RL training loop, GPU time for thousands of rollouts, Q16) that a prompted system doesn't — the decision should be gated on a measured comparison, following the same discipline used for other training-investment decisions in this bank (WebGPT's #39 Q15, REFRAG's #52 Q15):

```
1. Confirm data availability: do you have (or can you cheaply construct)
   a training set of (question, gold answer) pairs at sufficient scale
   (thousands, not dozens) for RL training to have enough signal? If not,
   this gate fails immediately regardless of projected quality gains.

2. Baseline: measure accuracy of a well-prompted Agentic RAG system
   (file 04) using your best available frontier model on a held-out
   eval set drawn from your actual production query distribution.

3. Prototype: RL-train a smaller open-weight model on a subset of your
   training data (a scaled-down run, not the full training budget) and
   measure the same eval set's accuracy, plus average search-call count
   (Q11) as an efficiency signal.

4. Project full-scale training cost and expected inference-cost savings
   at your production query volume (a smaller RL-trained model is
   typically cheaper per query to serve than repeatedly prompting a
   frontier model).

5. Gate: proceed to full training only if the prototype's accuracy
   trend is promising AND the projected inference-cost savings at your
   volume clear the training investment within an acceptable payback
   period -- exactly the framing Q13 uses to decide between the two
   approaches, now made quantitative.
```

The distinguishing consideration versus similar gates elsewhere in this bank is data availability as a hard precondition, not just a cost factor — unlike REFRAG or WebGPT-style training, which can bootstrap from existing production traffic or demonstrations, Search-R1's outcome-reward approach specifically needs verifiable gold answers, which many production QA tasks simply don't have ready-made at the volume RL training needs.

</details>

---

## Q16. What is the cost and infrastructure overhead of RL training rollouts at scale? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Unlike supervised fine-tuning's single forward-backward pass per training example, RL training requires running full multi-turn rollouts (Q2's rollout loop, potentially several `<search>` turns each involving a retrieval call and a generation call) for every training example, multiple times per prompt for GRPO's group-relative comparison (Q10, Q12) — this makes RL training substantially more compute- and infrastructure-intensive than standard fine-tuning for a comparable base model size.

Illustrative cost structure at a GRPO group size of 8 rollouts per prompt, average 3 search turns per rollout, across a training set of 10,000 prompts: this implies roughly 240,000 individual generation calls plus 240,000 retrieval calls just for one training epoch — a meaningfully larger compute footprint than the single pass per example a standard SFT run requires. Infrastructure overhead beyond raw compute: a rollout engine capable of interleaving generation and live retrieval calls during training (not just at inference), specialized RL training frameworks (veRL, OpenRLHF, TRL) to manage the PPO/GRPO update loop and loss masking (Q3), and — since rollouts depend on the *current* retriever's live behavior — a retrieval backend that can sustain training-time query volume, which is a different operational profile than a retriever only ever serving inference-time traffic. This infrastructure complexity is the primary reason the Q15 decision gate exists: the tooling investment is non-trivial even before accounting for the compute cost itself.

</details>

---

## Q17. What security and trust risks are specific to RL-trained search policies? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

- **Reward hacking exploited adversarially** — Q5's failure mode 1 (over/under-searching to game outcome reward) describes an unintentional training artifact, but the same mechanism is exploitable if an attacker can influence the training data or reward computation: a corrupted subset of (question, gold answer) pairs with subtly wrong gold answers could train the policy toward systematically incorrect behavior on that question class, and because the reward is purely outcome-based with no process supervision (Q4), there's no intermediate check that would catch a bad gold label before it shapes policy weights.
- **Frozen-retriever poisoning propagating through training** — if the retriever's index is poisoned during the RL training period (a malicious document inserted into the training-time corpus), the policy can learn to trust and route toward that poisoned content specifically, since outcome reward has no mechanism to distinguish "the model reasoned well and happened to retrieve bad evidence" from "the model reasoned well using good evidence" — both look identical from the reward function's perspective if the final answer happens to still be correct, and a policy trained partly on poisoned retrieval could generalize a preference for similar-looking-but-unreliable sources at inference time.
- **Opacity of the learned search strategy** — as with any RL-trained policy (WebGPT, #39 Q17), the model's query-phrasing and search-timing strategy is embedded in weights rather than inspectable in a prompt, making it harder to audit *why* the model chose to search (or not) in a specific production instance, and harder to rapidly patch a discovered bad behavior short of retraining (Q13's iteration-speed trade-off).
- **Training-time retrieval infrastructure as an attack surface** — since rollouts make live retrieval calls during training (Q16), the training-time retrieval backend is itself infrastructure that needs the same security posture as production retrieval (access controls, monitoring for anomalous query patterns), which is easy to under-invest in if a team treats "training infrastructure" as lower-stakes than "production infrastructure."

Mitigation: validate training data quality (gold-answer correctness) with the same rigor applied to any supervised training set, since RL's outcome-only signal provides no independent check on label quality; monitor the training-time corpus for the same poisoning risks as a production corpus; and treat a trained policy's behavior as requiring the same auditing discipline as WebGPT's (ongoing production transcript review, Q17 of #39) rather than assuming automatic outcome-reward training is inherently safer than human-preference RLHF.

</details>

---

## Q18. Design a Search-R1-style system for a biomedical literature research assistant. `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Requirements:** researchers need accurate answers to multi-hop biomedical questions (drug interactions, clinical trial outcomes) grounded in PubMed/clinical-trial literature; a small, cost-efficient model is preferred over repeatedly calling a large frontier model at research-team scale; retrieval quality against biomedical literature specifically (not general web content) is critical.

```
1. Training data: construct (question, gold answer) pairs from existing
   biomedical QA datasets and curated clinical-trial outcome data --
   this is DeepRetrieval's actual approach (Q5's family comparison),
   which trains small (3B) models to outperform GPT-4o/Claude-scale
   models specifically by optimizing retrieval recall against a
   biomedical corpus rather than a general one.

2. Retriever: a dense retriever fine-tuned on biomedical text (not a
   general-purpose embedding model) as the frozen backend (Q1) -- this
   directly addresses the frozen-retriever blind-spot risk (Q14) by
   ensuring the retrieval ceiling itself is already domain-appropriate
   before RL training even begins.

3. Reward: outcome-based EM/F1 against curated gold answers (Q4),
   supplemented with a retrieval-recall component for the query-
   generation sub-task specifically (following DeepRetrieval's variant,
   Q5's family table) given that in this domain, retrieval quality
   itself (not just final-answer correctness) is a measurable and
   important intermediate signal.

4. Multi-turn handling (Q2): biomedical questions often require
   chaining evidence across multiple papers (a drug's mechanism from
   one source, an interaction study from another) -- max_turns tuned
   generously (Q10) given this domain's genuine multi-hop depth needs.

5. Monitoring (Q14): segment accuracy by search-skip rate specifically
   for rare/novel drug or condition names, where a frozen retriever's
   coverage gaps are most likely to concentrate, and route detected
   gaps back into retriever improvement rather than further policy
   training.
```

The key design choice, following DeepRetrieval's actual precedent, is treating retrieval quality itself as a first-class, separately-optimized concern rather than purely trusting outcome reward to implicitly shape good search behavior — appropriate specifically because this domain's stakes (clinical decision support) justify the extra complexity of a retrieval-quality reward component that a lower-stakes general QA task might skip.

</details>

---

## Q19. What happens when the reward signal is noisy or based on mislabeled gold answers, and how do you debug it? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Since Search-R1's entire training signal is outcome correctness against a gold answer (Q4), a training set containing mislabeled or ambiguous gold answers directly corrupts the policy's learning target — the model can be systematically "punished" for correct reasoning that happens to disagree with a wrong gold label, or "rewarded" for incorrect reasoning that happens to coincidentally match a wrong label, with no process-level check to catch either case (unlike a system with intermediate supervision that could flag a discrepancy earlier in the trajectory).

**Symptom:** training plateaus below expected accuracy, or the policy exhibits confident, well-reasoned trajectories (readable `<think>` traces that look sound) landing on answers that get zero reward — a strong signal to manually audit a sample of zero-reward-despite-good-reasoning trajectories specifically, since this pattern is much more likely to indicate gold-label noise than genuine policy failure (a policy failing due to its own bad reasoning typically produces visibly weaker `<think>` traces, not confident-and-wrong ones). **Debugging:** (1) sample and manually verify gold answers for training examples where the policy's answer format is well-formed and its reasoning trace looks sound but reward was still zero; (2) check for systematic label-format mismatches (an exact-match reward function penalizing a correct answer phrased slightly differently than the gold string, e.g. "1919" vs. "the year 1919") which is a normalization bug in the reward function rather than a true label error, and just as damaging to training; (3) if genuine label noise is confirmed at meaningful scale, either clean the training set or switch to a more permissive matching criterion (F1 rather than strict EM, or an LLM-judge-based reward) that's more robust to superficial answer-format variation.

</details>

---

## Q20. What are the limitations of Search-R1-style RL-trained search, and how might the field evolve? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Current limitations: (1) **requires large-scale gold-answer training data** (Q15) — a hard precondition that rules out many real production tasks lacking this kind of labeled dataset; (2) **frozen retriever caps achievable quality** (Q5, Q14) — no amount of policy training fixes a fundamentally weak retrieval backend, only teaches the policy to route around its gaps; (3) **outcome-only reward provides no process supervision** (Q4, Q19) — this is what makes training cheap and self-supervising, but also means gold-label noise, reward hacking, and reasoning-quality issues have no intermediate checkpoint to be caught at, only the final answer; (4) **training infrastructure and compute cost are substantially higher than supervised fine-tuning** (Q16), and higher than a purely prompted alternative's near-zero setup cost.

Likely evolution: **joint training of retriever and policy** (rather than treating the retriever as permanently frozen, Q1) is a natural extension addressing limitation (2) directly, letting the retrieval backend itself improve based on which of its outputs actually led to correct answers, not just the query-phrasing policy; **hybrid reward signals** combining outcome correctness with lightweight process signals (retrieval-quality reward, as DeepRetrieval already does for the query-generation sub-task, Q5's family table) to reduce the credit-assignment difficulty of pure outcome reward (Q12); and continued growth of the broader RL-for-reasoning recipe (the DeepSeek-R1 lineage Search-R1 builds on, Q8) extending beyond search specifically into other tool-use decisions, treating "when to call any external tool" as one unified learned decision rather than a family of separately-trained, tool-specific policies.

</details>

---

## Real-World Applications

- **Open-domain multi-hop QA systems**: Search-R1-style training reported large gains (+41% Qwen2.5-7B, +20% Qwen2.5-3B) over prompted RAG baselines on seven QA benchmarks (HotpotQA, 2WikiMultihopQA, Musique, etc.)
- **Literature/biomedical search agents**: DeepRetrieval trains small (3B) models that outperform GPT-4o/Claude on PubMed and clinical-trial literature retrieval tasks by directly optimizing recall
- **Cost-sensitive search agents**: RL-trained policies that learn to minimize the number of search calls per query are attractive where each search call has a real dollar cost (paid search APIs)
- **Autonomous research assistants**: sibling frameworks (R1-Searcher, ReSearch) demonstrate the same RL-for-search recipe generalizing to reflection and self-correction behavior without hand-authored heuristics
