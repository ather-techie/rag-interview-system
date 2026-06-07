# 08 — Speculative RAG

> A smaller specialist model generates multiple draft answers; a larger generalist model selects and refines the best one — balancing quality and latency.

---

## Q1. What is Speculative RAG and what is the core insight behind it? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Speculative RAG** (Wang et al., 2024) is inspired by speculative decoding in LLMs. The core insight is:

> A small, specialized model is faster and often better at *drafting* answers over retrieved documents. A large, general model is better at *judging and refining* those drafts.

**Pipeline:**

1. Retrieved documents are **partitioned into subsets**.
2. A **small specialist RAG drafter** generates one candidate answer per subset.
3. A **large generalist LLM** receives all candidate answers (not the raw documents) and selects/refines the best one.

This reduces the number of tokens the large LLM must process — instead of reading all retrieved chunks, it only reads short candidate answers — improving both **quality** and **latency**.

```
Retrieved chunks (k=10)
  │
  ├── Subset 1 (2 chunks) ─► Small RAG drafter ─► Candidate answer 1 + rationale
  ├── Subset 2 (2 chunks) ─► Small RAG drafter ─► Candidate answer 2 + rationale
  ├── Subset 3 (2 chunks) ─► Small RAG drafter ─► Candidate answer 3 + rationale
  ├── Subset 4 (2 chunks) ─► Small RAG drafter ─► Candidate answer 4 + rationale
  └── Subset 5 (2 chunks) ─► Small RAG drafter ─► Candidate answer 5 + rationale
                                                          │
                                        ┌─────────────────┘
                                        ▼
                           Large generalist LLM
                           (reads only candidates, not raw chunks)
                                        │
                                        ▼
                                   Final Answer
```

</details>

---

## Q2. How does document partitioning work in Speculative RAG? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

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

</details>

---

## Q3. What is the role of the large generalist LLM in Speculative RAG? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

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

</details>

---

## Q4. How does Speculative RAG compare to Speculative Decoding in standard LLM inference? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

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

</details>

---

## Q5. When would Speculative RAG outperform Agentic RAG, and when would it not? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

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

</details>

---

## Q6. How do you select and fine-tune the small drafter model in Speculative RAG? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The drafter model is typically a smaller version (or different model) specialized for fast, accurate answer extraction from limited context.

**Selection criteria:**

| Criterion | Recommendation |
|-----------|----------------|
| **Base model** | Llama 2 7B, Mistral 7B, or fine-tuned domain expert |
| **Latency target** | Should answer in <100ms per partition (on single GPU) |
| **Quality bar** | Answer accuracy >80% on validation set (can be loose; verifier refines) |
| **Fine-tuning** | Improves 3-5pp if trained on domain Q&A pairs |

**Fine-tuning recipe:**

```python
# 1. Prepare training data (domain Q&A pairs + retrieved context)
train_data = [
    {
        "question": "What is the company revenue?",
        "context": "Q3 2024 revenue was $10M, up 20%...",
        "answer": "Q3 2024 revenue: $10M",  # Extractive answer
        "rationale": "Extracted from financial report"
    },
    ...
]

# 2. Fine-tune using SFT (Supervised Fine-Tuning)
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B")

# Format data
formatted_data = [
    f"Question: {ex['question']}\nContext: {ex['context']}\nAnswer: {ex['answer']}"
    for ex in train_data
]

# Train
training_args = TrainingArguments(
    output_dir="drafter_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=formatted_data,
)
trainer.train()

# 3. Evaluate on validation set
# Metric: Answer exact match or BLEU against gold answers
```

**Typical improvements:**
- Base Mistral 7B: ~70% answer accuracy on in-domain data.
- +3pp fine-tuning on 5K examples.
- +2pp quantization (4-bit) without quality loss.

</details>

---

## Q7. Implement the Speculative RAG pipeline end-to-end in Python `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Here's a complete implementation:

```python
import asyncio
from typing import List
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

class SpeculativeRAG:
    def __init__(self, drafter_model="mistral-7b", verifier_model="gpt-4o-mini"):
        self.drafter = ChatOpenAI(model=drafter_model, temperature=0)
        self.verifier = ChatOpenAI(model=verifier_model, temperature=0)
        self.vectorstore = None
    
    def partition_documents(self, documents: List[str], num_subsets: int = 5) -> List[List[str]]:
        """Partition retrieved documents into subsets."""
        subset_size = max(1, len(documents) // num_subsets)
        partitions = [
            documents[i:i+subset_size]
            for i in range(0, len(documents), subset_size)
        ]
        return partitions[:num_subsets]  # Ensure exactly num_subsets
    
    async def draft_answer(self, query: str, documents: List[str]) -> dict:
        """Small model generates answer from one subset."""
        context = "\n".join(documents)
        prompt = f"""Question: {query}

Context:
{context}

Provide a concise answer with a brief rationale for your answer.

Answer: <answer>
Rationale: <why this is the best answer given the context>"""
        
        response = await asyncio.to_thread(self.drafter.invoke, prompt)
        
        # Parse response
        answer = response.content.split("Answer:")[1].split("Rationale:")[0].strip()
        rationale = response.content.split("Rationale:")[1].strip()
        
        return {
            "answer": answer,
            "rationale": rationale,
            "documents": documents
        }
    
    async def draft_all_candidates(self, query: str, documents: List[str], 
                                   num_drafters: int = 5) -> List[dict]:
        """Generate candidate answers in parallel."""
        partitions = self.partition_documents(documents, num_drafters)
        
        # Parallel drafting
        tasks = [self.draft_answer(query, part) for part in partitions]
        candidates = await asyncio.gather(*tasks)
        
        return candidates
    
    def verify_and_select(self, query: str, candidates: List[dict]) -> dict:
        """Large model selects and refines the best candidate."""
        
        candidates_text = "\n\n".join([
            f"Candidate {i+1}:\nAnswer: {c['answer']}\nRationale: {c['rationale']}"
            for i, c in enumerate(candidates)
        ])
        
        selection_prompt = f"""Question: {query}

Below are multiple candidate answers from different document subsets:

{candidates_text}

Task:
1. Evaluate which candidate is most accurate and well-supported.
2. If beneficial, synthesize a better answer from multiple candidates.
3. Provide your final answer.

Final Answer: <answer>
Reasoning: <explanation of which candidate(s) you selected and why>"""
        
        response = self.verifier.invoke(selection_prompt)
        
        # Parse response
        final_answer = response.content.split("Final Answer:")[1].split("Reasoning:")[0].strip()
        reasoning = response.content.split("Reasoning:")[1].strip()
        
        return {
            "final_answer": final_answer,
            "reasoning": reasoning,
            "num_candidates_considered": len(candidates)
        }
    
    async def query(self, query: str, k: int = 10) -> dict:
        """End-to-end pipeline."""
        
        # 1. Retrieve documents
        documents = self.vectorstore.similarity_search(query, k=k)
        doc_texts = [doc.page_content for doc in documents]
        
        # 2. Draft candidate answers in parallel
        candidates = await self.draft_all_candidates(query, doc_texts, num_drafters=5)
        
        # 3. Verify and select
        result = self.verify_and_select(query, candidates)
        
        return result

# Usage
rag = SpeculativeRAG()
rag.vectorstore = Chroma(...)

result = asyncio.run(rag.query("What is the company revenue?"))
print(result["final_answer"])
```

</details>

---

## Q8. How do you tune the number of subsets (m) and subset size in Speculative RAG? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The number of partitions `m` trades off drafter parallelism, verification cost, and answer diversity.

```
Quality vs. Latency for different m values:

        Quality (Answer accuracy)
             ↑
           0.88 │           m=3  m=5  m=7
                │          ╱    ╱    ╱ (more drafts = better)
           0.85 │   ──────╱────╱────╱
                │  m=1           
           0.80 │ (no partitioning)
                │
                └────────────────────────► Latency
                 100ms  200ms  400ms  600ms

           m=1 (no partitioning): lowest latency, lower quality
           m=5 (balanced):        medium latency, good quality
           m=10 (exhaustive):     high latency, marginal quality gain
```

**Tuning approach:**

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class SpeculativeRAGConfig:
    num_subsets: int
    drafter_model: str
    verifier_model: str
    
class GridSearchTuning:
    def __init__(self, validation_set):
        self.validation_set = validation_set  # 100 test queries
    
    def evaluate_config(self, config: SpeculativeRAGConfig):
        """Measure quality and latency for this config."""
        
        accuracies = []
        latencies = []
        
        for query, gold_answer in self.validation_set:
            import time
            start = time.time()
            
            result = asyncio.run(rag.query(query, k=10))
            
            latency = time.time() - start
            
            # Compute accuracy (e.g., BLEU or exact match)
            accuracy = compute_similarity(result["final_answer"], gold_answer)
            
            accuracies.append(accuracy)
            latencies.append(latency)
        
        return {
            "config": config,
            "avg_accuracy": np.mean(accuracies),
            "p95_latency": np.percentile(latencies, 95),
            "cost_per_query": estimate_cost(config)
        }
    
    def find_pareto_frontier(self, configs):
        """Find configs on Pareto frontier (highest accuracy at each latency level)."""
        
        results = [self.evaluate_config(cfg) for cfg in configs]
        
        # Sort by latency
        results.sort(key=lambda x: x["p95_latency"])
        
        pareto_frontier = []
        max_accuracy = 0
        
        for result in results:
            if result["avg_accuracy"] >= max_accuracy:
                pareto_frontier.append(result)
                max_accuracy = result["avg_accuracy"]
        
        return pareto_frontier

# Example: Try m=3, 5, 7, 10
configs = [
    SpeculativeRAGConfig(num_subsets=m, drafter_model="mistral-7b", 
                        verifier_model="gpt-4o-mini")
    for m in [3, 5, 7, 10]
]

tuner = GridSearchTuning(validation_set)
pareto = tuner.find_pareto_frontier(configs)

# Output:
# m=3: 85% accuracy, 120ms latency
# m=5: 87% accuracy, 250ms latency ← Recommended (good accuracy/latency tradeoff)
# m=7: 87.5% accuracy, 380ms latency
# m=10: 88% accuracy, 600ms latency (too slow)
```

</details>

---

## Q9. How does Speculative RAG behave when no retrieved document is relevant? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A failure mode: all retrieved documents are off-topic. Speculative RAG can't save a bad retrieval round.

```
Retrieved docs:
├─ [Document 1] (irrelevant)
├─ [Document 2] (irrelevant)
└─ [Document 3] (irrelevant)

→ All drafters generate hallucinations or "I don't know"
→ Verifier picks "best of bad options"
→ Result: Hallucinated answer presented as confident
```

**Mitigation strategies:**

| Strategy | Implementation | Trade-off |
|----------|----------------|-----------|
| **Confidence threshold** | If all candidates score <0.5, abstain ("I don't know") | May miss valid answers from weak evidence |
| **Fallback to web search** | Detect all-irrelevant case, trigger web search | +500ms latency, higher cost |
| **Semantic diversity check** | If all candidates are too similar, flag for review | Human-in-the-loop latency |
| **Retrieval quality gating** | Filter retrieved docs before partitioning (CRAG style) | Adds evaluator overhead |

```python
class RobustSpeculativeRAG(SpeculativeRAG):
    def detect_retrieval_failure(self, candidates: List[dict], 
                               query: str, threshold=0.4) -> bool:
        """Check if retrieval was likely unsuccessful."""
        
        # Heuristic: If all drafts are very short or contain "I don't know"
        abstentions = sum(1 for c in candidates 
                         if len(c["answer"]) < 20 or "don't know" in c["answer"])
        
        if abstentions / len(candidates) > 0.6:
            return True  # Likely retrieval failure
        
        # Semantic similarity: if all candidates are nearly identical, 
        # no new information from partitioning
        similarities = []
        for i in range(len(candidates)):
            for j in range(i+1, len(candidates)):
                sim = cosine_similarity(candidates[i]["answer"], 
                                      candidates[j]["answer"])
                similarities.append(sim)
        
        if all(sim > 0.9 for sim in similarities):
            return True  # All drafts redundant
        
        return False
    
    async def query_with_fallback(self, query: str, k: int = 10) -> dict:
        """Query with web search fallback."""
        
        # 1. Retrieve documents
        documents = self.vectorstore.similarity_search(query, k=k)
        doc_texts = [doc.page_content for doc in documents]
        
        # 2. Draft candidates
        candidates = await self.draft_all_candidates(query, doc_texts, num_drafters=5)
        
        # 3. Detect failure
        if self.detect_retrieval_failure(candidates, query):
            print("Retrieval quality low; falling back to web search")
            web_results = web_search_api.search(query, num_results=k)
            web_texts = [r["snippet"] for r in web_results]
            candidates = await self.draft_all_candidates(query, web_texts, num_drafters=5)
        
        # 4. Verify
        result = self.verify_and_select(query, candidates)
        return result
```

</details>

---

## Q10. What are the GPU/memory requirements for serving Speculative RAG in production? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Speculative RAG runs two models in parallel, each with different resource requirements.

**Resource table (single request):**

| Component | Model Size | VRAM | Batch Size | Total VRAM |
|-----------|-----------|------|-----------|-----------|
| Drafter | Mistral 7B | 14GB | 5 parallel | 14GB (shared across 5 subsets) |
| Verifier | GPT-4o-mini | API | N/A | N/A (or 8GB if self-hosted) |
| Vector DB | Chroma (in-memory) | ~2GB (for 100K embeddings) | N/A | 2GB |

**Single-GPU setup:**

```
H100 GPU (80GB VRAM):
├─ Drafter (14GB)
├─ Verifier/Router (4GB)
├─ Vector DB (2GB)
├─ Batch buffers (20GB)
└─ Headroom (40GB) ← Can serve ~4 concurrent requests

Recommended: H100 for >100 QPS; A100 (40GB) for <50 QPS
```

**Multi-GPU deployment (production):**

```
┌────────────────────────────────────┐
│  Load Balancer (incoming requests) │
└───────────┬────────────────────────┘
            │
    ┌───────┴───────┐
    ▼               ▼
[GPU 0: Drafter]  [GPU 1: Drafter]  (parallelism)
Mistral 7B        Mistral 7B
14GB each         14GB each
    │               │
    └───────┬───────┘
            ▼
   [GPU 2: Verifier]
   GPT-4o or Llama 70B
   16-40GB
            │
            ▼
   [Vector DB Server]
   Qdrant or Milvus
   (separate machine, CPU-based)
```

**Latency and throughput:**

```python
class SpeculativeRAGServer:
    def __init__(self, num_drafter_gpus=2, num_verifier_gpus=1):
        self.drafter_pool = GPUPool(num_drafter_gpus, "mistral-7b")
        self.verifier = GPUPool(num_verifier_gpus, "gpt-4o-mini")
        self.vectordb = RemoteVectorDB("qdrant://localhost:6333")
    
    def estimate_latency(self, k=10, num_subsets=5):
        """Estimate end-to-end latency."""
        
        # Retrieval: 10-50ms
        retrieval_time = 30
        
        # Drafting: parallel across num_subsets, but sequential per subset
        # Mistral 7B → ~50ms per subset (average)
        drafting_time = (num_subsets / self.drafter_pool.num_gpus) * 50
        
        # Verification: ~100ms
        verification_time = 100
        
        # Total
        total = retrieval_time + max(drafting_time, 50) + verification_time
        return total  # ~180-250ms expected
    
    def estimate_throughput(self, qps_budget=100):
        """How many concurrent requests can we serve?"""
        
        avg_latency_ms = self.estimate_latency()
        max_concurrent = qps_budget * (avg_latency_ms / 1000)  # Little's law
        
        return max_concurrent  # e.g., 100 QPS × 0.2s = 20 concurrent

# Example tuning
server = SpeculativeRAGServer(num_drafter_gpus=2, num_verifier_gpus=1)
print(f"Estimated latency: {server.estimate_latency()}ms")
print(f"Max throughput at 100 QPS: {server.estimate_throughput(100)} concurrent")
```

**Cost optimization:**

1. **Batch inference** — Group requests, run drafters in parallel.
2. **Quantization** — 4-bit Mistral 7B → 7GB VRAM (vs. 14GB), <5% accuracy loss.
3. **LoRA adapters** — Multiple domain-specific drafters sharing base weights.
4. **Caching** — Cache candidate answers for repeated queries.

**Recommended deployment:**

| Use Case | Setup | Cost/Month |
|----------|-------|-----------|
| <50 QPS | H100 (1 GPU) + API verifier | ~$3K |
| 50-200 QPS | 2× A100 (drafter + verifier) | ~$10K |
| >200 QPS | 4× H100 (2 drafters + 1 verifier + 1 reserve) | ~$20K |


</details>
---

## Q11. How do you architect and orchestrate a multi-GPU serving cluster for Speculative RAG to maximize drafter parallelism while keeping verifier utilization high? [Intermediate]

<details>
<summary>?? Show Answer</summary>

**Answer:**

Speculative RAG pairs a fast drafter and slower verifier. The bottleneck is keeping the verifier busy while drafters compete for its attention.

**Architecture for high utilization:**

Use 2 drafters on A100s, 1 verifier on H100, batch verification of 2 drafts at a time. This keeps verifier utilization >80%.

**Optimization strategies:**

1. **Batch verification** - Verifier processes 2-4 drafts in parallel.
2. **Token-level speculative decoding** - Verifier checks draft token-by-token for early termination.
3. **Adaptive draft length** - Shorten drafts if verifier is busy (monitors queue depth).
4. **Speculative token bundling** - Group tokens into packets for faster verification.

**Monitoring:**

- Drafter GPU utilization (target: >80%).
- Verifier GPU utilization (target: >80%).
- Queue depth (draft_queue and verify_queue).
- Latency P95 (target: <2s for interactive use).

</details>

---

## Q12. How can adversarial inputs exploit systematic disagreement between the drafter and verifier models, and what detection and mitigation strategies prevent this from degrading answer quality? [Advanced]

<details>
<summary>?? Show Answer</summary>

**Answer:**

**Attack: Drafter-verifier disagreement exploitation**

Attacker crafts queries that cause drafter and verifier to systematically disagree, allowing malicious tokens to slip through verification.

**Defences:**

1. **Disagreement monitoring** - Flag queries with disagreement rate >30%; escalate to human.

2. **Ensemble verification** - Use multiple independent verifiers; require consensus on contentious tokens.

3. **Adversarial input detection** - Detect suspicious patterns (rare tokens, incoherent wording) before inference.

4. **Token-level rollback** - If verifier detects error, roll back to last verified token and regenerate.

5. **Confidence calibration** - Return lower confidence for high-disagreement queries.

6. **Drafter robustness training** - Fine-tune drafter on adversarial examples to agree with verifier.

7. **Continuous retraining** - Periodically retrain both models to stay synchronized.

**Defence-in-depth:** Attackers must defeat multiple layers; single-layer attacks fail.

</details>
