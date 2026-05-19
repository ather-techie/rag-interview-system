# 07 — Self-RAG

> The LLM is trained to reflect on its own outputs — deciding whether to retrieve, critiquing retrieved passages, and rating its own generation for faithfulness.

---

## Q1. What is Self-RAG and how does it differ from standard RAG? `[Basic]`

**Answer:**

**Self-RAG** (Asai et al., 2023) fine-tunes an LLM to control its own retrieval and generation through special **reflection tokens**:

| Reflection Token | Meaning |
|---|---|
| `[Retrieve]` | Should I retrieve for this generation step? |
| `[IsRel]` | Is this retrieved passage relevant? |
| `[IsSup]` | Does my generation faithfully use the passage? |
| `[IsUse]` | Is my overall response useful to the user? |

**Key difference from standard RAG:** In standard RAG, retrieval always happens and the LLM has no say. In Self-RAG, the model itself decides:
- *Whether* to retrieve (some generations don't need retrieval)
- *Which* retrieved passages to use
- *How good* its own generation is

This results in more adaptive, higher-quality outputs — at the cost of requiring fine-tuning.

---

## Q2. What are the four reflection token types in Self-RAG and what do they control? `[Intermediate]`

**Answer:**

1. **`[Retrieve]`** — Generated before each segment. If the model outputs `[Retrieve]=Yes`, the system fetches documents. If `No`, it generates from its own knowledge.

2. **`[IsRel]` (Relevance)** — Generated after each retrieved passage is shown. Scores `[Relevant]` or `[Irrelevant]`. Irrelevant passages are excluded from context.

3. **`[IsSup]` (Support)** — Generated after each output segment. Scores `[Fully supported]`, `[Partially supported]`, or `[No support]`. Measures factual grounding.

4. **`[IsUse]` (Utility)** — Generated at the end of the full response. Scores utility on a 1–5 scale. Used for candidate selection if multiple generations are sampled.

Together, these tokens let the model perform **inference-time tree search** — generate multiple candidate continuations and select the best by combining the reflection scores.

---

## Q3. How is Self-RAG trained? `[Intermediate]`

**Answer:**

Self-RAG requires a **two-stage training pipeline**:

**Stage 1 — Create training data:**
1. Take a standard instruction-following dataset.
2. Use a **critic LLM** (e.g., GPT-4) to retroactively annotate each (instruction, response) pair with reflection tokens.
3. For segments that needed retrieval, insert actual retrieved passages and annotate `[IsRel]`, `[IsSup]`, `[IsUse]`.

**Stage 2 — Fine-tune the generator:**
- Fine-tune a base LLM (e.g., Llama 2 7B/13B) on the augmented dataset using standard causal language modeling.
- The model learns to generate reflection tokens as natural continuations.
- No separate reward model is needed — reflection tokens are part of the vocabulary.

**Result:** A single model that does both retrieval gating and generation quality assessment.

---

## Q4. How does Self-RAG use reflection tokens at inference time to select the best output? `[Advanced]`

**Answer:**

At inference time, Self-RAG performs a **segment-level beam search**:

1. For each generation segment, sample multiple continuations.
2. Score each continuation using its reflection token combination:
   - Prefer `[IsSup]=Fully supported` over `[IsSup]=No support`
   - Prefer higher `[IsUse]` scores
   - Weight scores using a tunable parameter `α`
3. Select the highest-scoring continuation and proceed to the next segment.

**Final score formula (simplified):**
```
score = α × P(IsSup=Fully supported) + (1-α) × P(IsUse=5)
```

This makes Self-RAG controllable at inference time — increasing `α` emphasizes factuality; decreasing it emphasizes overall usefulness.

---

## Q5. What are the practical limitations of Self-RAG for production deployment? `[Advanced]`

**Answer:**

| Limitation | Impact | Workaround |
|---|---|---|
| **Requires fine-tuning** | Can't use closed-source models (GPT-4, Claude) | Use CRAG or prompted judges as proxies |
| **Training data cost** | Critic LLM annotation is expensive | Limit to high-value domains |
| **Inference overhead** | Multiple generation candidates + reflection scoring | Reduce beam width; use greedy for low-stakes queries |
| **Outdated after training** | Reflection thresholds baked into weights | Fine-tune periodically or allow runtime threshold overrides |
| **Smaller base models** | Self-RAG was trained on 7B/13B models | Quality degrades for very complex reasoning |

**Bottom line:** Self-RAG is most appropriate for specialized, high-accuracy domains (medical, legal) where the cost of fine-tuning is justified and factual grounding is critical. For general-purpose chatbots, prompted evaluation (CRAG-style) is more practical.
