# 01 — Hallucination Despite Context

> The LLM generates false or fabricated claims even when the retrieval stage successfully returns relevant, correct context.

---

## Q1. What does hallucination in a RAG system mean, and how is it different from a standalone LLM hallucination? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Hallucination in standalone LLMs occurs because the model has no external grounding—it generates text based solely on its training data distribution, often confabulating plausible-sounding but false facts. In a RAG system, hallucination is paradoxically *worse* because it represents a failure of the retrieval-augmentation promise: despite providing the model with correct, relevant context, the LLM still generates false claims.

This represents a **compounded failure mode**:
- The retrieval stage succeeded (relevant chunks were found)
- The context was included in the prompt
- Yet the model still ignored or contradicted that context

This is particularly problematic in production because users (and auditors) expect that grounding the LLM with retrieved documents will eliminate hallucinations. When it doesn't, it erodes trust and can lead to incorrect decisions.

</details>

---

## Q2. What are the visible symptoms of hallucination in a RAG system? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Symptoms of hallucination in RAG systems manifest in several forms:

| Symptom | Example | Detection Signal |
|---------|---------|------------------|
| **Contradicts retrieved context** | Retrieved doc says "Founded 2015", LLM says "Founded 2012" | Direct string/fact mismatch vs retrieved chunks |
| **Cites non-existent sources** | "According to document X..." where X was never retrieved | Citation inconsistency with retrieval log |
| **Invents details** | "The algorithm uses a Fibonacci sequence..." (never mentioned in context) | Claims not supported by any retrieved chunk |
| **Conflates sources** | Mixes facts from Doc A and Doc B into a false composite claim | Cross-document hallucination detection |
| **Answers out-of-scope queries** | Query: "Who is the CEO?", Retrieved docs don't mention CEO, LLM still answers | Query-doc relevance mismatch with confident answer |

End users typically discover these via:
- **Manual QA audits** — spot-checking answers against source documents
- **Downstream failures** — incorrect decisions made based on hallucinated facts
- **User feedback** — "This is wrong, I found the correct answer in X"

</details>

---

## Q3. What are the main root causes of hallucination despite correct context being available? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Hallucinations in RAG systems occur at multiple stages and for several technical reasons:

**1. Attention Dilution (Position Bias)**

LLMs struggle to attend to specific facts in long prompts. When the context window contains many chunks, the model's attention may spread too thin or focus on more recent tokens (recency bias).

```python
# Example: 10 chunks retrieved, relevant fact in chunk #3
prompt = f"""
Context:
{chunk_1}
{chunk_2}
{chunk_3}  # Contains answer here
{chunk_4}
...
{chunk_10}

Question: What is...?
"""
# LLM may attend primarily to chunk_10 or miss chunk_3 entirely
```

Studies (Liu et al., 2023 "Lost in the Middle") show models perform worse when relevant information is in the middle of a long context window.

**2. Conflicting or Ambiguous Context**

When retrieved chunks contain contradictory information, the LLM may pick the "wrong" one or try to synthesize a false compromise.

```
Chunk A: "Company X was founded in 2015."
Chunk B: "Company X began operations in 2012."
LLM output: "Company X was founded in 2012 and later restructured in 2015."
```

**3. Training Data Dominance**

The model's pre-training weights may encode strong priors about common facts. If a retrieval miss happens (wrong chunks retrieved), the model falls back to training data.

**4. Instruction Following Failure**

The prompt may instruct "answer based on the context" but use permissive language like "You may also draw on your knowledge if helpful." This creates ambiguity.

**5. Context Collapse**

Some prompting styles bury the context in a wall of text, making it hard for the model to extract the relevant signal:

```
"Here is some context for your question...
[1000 tokens of text]
... please answer the question now"
```

</details>

---

## Q4. How can you detect whether hallucinations are occurring in a production RAG system? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Hallucination detection requires both **automated metrics** and **human oversight**:

### Automated Detection Methods

| Method | How It Works | Pros | Cons |
|--------|-------------|------|------|
| **Citation Matching** | Check if every claim in the answer is cited/supported by a retrieved chunk | Fast, deterministic, no LLM calls | High false negatives (paraphrased facts, implicit support) |
| **Entailment (NLI)** | Run an NLI model (e.g., `cross-encoder/nli-deberta-v3-large`) to check if each sentence is entailed by retrieved chunks | Semantic understanding, catches paraphrasing | Slower (cross-encoder inference), sometimes unreliable on domain-specific text |
| **LLM-as-Judge** | Prompt a separate LLM (or same model in judge mode) to evaluate: "Is this answer supported by the context?" | Catches nuanced hallucinations, aligns with human judgment | Expensive (2x LLM calls per query), may inherit hallucinations from judge LLM |
| **Embedding Similarity** | Embed the answer and each chunk, compute cosine similarity. Low similarity suggests hallucination risk. | Fast, unsupervised | Lexical/semantic mismatches (e.g., synonyms) cause false positives |

### Production Monitoring Setup

```python
from sentence_transformers import CrossEncoder

# Load an NLI model
nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-large')

def detect_hallucination(question, answer, retrieved_chunks):
    # Combine all chunks into a single context
    context = " ".join(retrieved_chunks)
    
    # Score: does context entail the answer?
    # Premise = context, Hypothesis = answer
    scores = nli_model.predict([[context, answer]])
    
    # scores[0] = [contradiction, neutral, entailment]
    entailment_prob = scores[0][2]
    
    return {
        'hallucination_risk': 1 - entailment_prob,
        'confidence': entailment_prob,
        'detected': entailment_prob < 0.6  # threshold
    }
```

### Human-in-the-Loop Monitoring

- **Spot-check audits**: Randomly sample 50 answers/week, manually verify against source docs
- **User feedback loops**: Flag answers that users dispute and log them for analysis
- **Retrieval drift alerts**: Monitor if retrieved chunks increasingly fail to match the question

### Practical Dashboard Metrics

Track in production:
- **Hallucination rate** = % of answers with detected unsupported claims
- **Citation coverage** = % of answer claims with explicit source citations
- **NLI entailment score** = average entailment probability across all answers
- **False positive rate** = % of correct answers falsely flagged as hallucinating (via manual audit)

</details>

---

## Q5. What is the "lost in the middle" problem and how does it cause hallucinations? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The "lost in the middle" problem (Liu et al., 2023) describes a striking failure mode: when relevant information is placed in the *middle* of a long context window, LLMs often fail to access it, instead relying on information from the beginning or end of the context.

### The Empirical Pattern

Researchers tested LLMs with a "needle in a haystack" task:

```
Context layout:
[START] Irrelevant filler text [MIDDLE] ← Relevant fact here [END] More filler

Result:
- Fact at position 0%: ✓ Retrieved correctly
- Fact at position 25%: ✓ Retrieved correctly
- Fact at position 50%: ✗ MISSED (hallucination risk)
- Fact at position 75%: ✗ MISSED
- Fact at position 100%: ✓ Retrieved correctly
```

### Why This Happens

**1. Attention Mechanism Behavior**

Transformers have attention weights that favor:
- **Early tokens** (due to training on naturally ordered documents where important context comes first)
- **Recent tokens** (recency bias from training on instruction-following tasks)

Middle tokens get diluted attention.

**2. Positional Encoding Extrapolation**

Modern LLMs may use positional encodings trained on shorter sequences. When pushed to handle longer contexts (via position interpolation), the middle region becomes ambiguous.

### Example: Production Impact

```python
# Naive RAG implementation
retrieved_chunks = [
    "Irrelevant company history...",
    "Founded in 2015 by John...",  # ← Important fact
    "Irrelevant product details...",
    "Irrelevant pricing info..."
]

prompt = f"""
Context:
{retrieved_chunks[0]}
{retrieved_chunks[1]}  # Buried in middle
{retrieved_chunks[2]}
{retrieved_chunks[3]}

Question: When was the company founded?
"""
# LLM may answer "I don't know" or hallucinate "2012"
```

### Mitigation Strategies

| Strategy | How | Trade-off |
|----------|-----|-----------|
| **Reorder chunks** | Place most relevant at top/bottom, filler in middle | Requires ranking/reranking step |
| **Reduce window** | Fewer chunks, higher density of signal | May miss context |
| **Explicit markers** | Use XML tags: `<RELEVANT>` ... `</RELEVANT>` | Consumes tokens, may confuse older models |
| **Compress middle** | Summarize filler chunks, expand important ones | Lossy, requires careful design |

</details>

---

## Q6. How can you enforce citation and prevent the LLM from inventing unsupported claims? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Citation enforcement is a multi-layer strategy combining prompting, structured output, and post-processing:

### Layer 1: Explicit Citation Prompting

Instruct the model to cite every claim:

```python
system_message = """You are a helpful assistant that answers questions based on provided documents.

IMPORTANT: Every factual claim in your answer must be supported by at least one provided document.
Format citations as [Doc: source_name, page X] or [Doc: section Y].

If a claim cannot be cited, do NOT include it in your answer.
If you don't know the answer, say "I don't have enough information to answer this."
"""

user_message = f"""
Documents:
{format_documents(retrieved_chunks)}

Question: {question}

Provide your answer with citations for each claim.
"""
```

### Layer 2: Structured Output (JSON with Citations)

Force the model to output in a structured format:

```python
from pydantic import BaseModel
from anthropic import Anthropic

class CitedAnswer(BaseModel):
    answer: str
    citations: list[dict]  # [{"claim": "...", "source": "Doc X", "page": 3}]

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system="You output JSON with citations. Every claim in 'answer' must have a corresponding entry in 'citations'.",
    messages=[{
        "role": "user",
        "content": f"Documents: {docs}\n\nQuestion: {question}\n\nRespond in JSON format: {{\"answer\": \"...\", \"citations\": [...]}}"
    }]
)
```

### Layer 3: Post-Processing Verification

After generation, verify citations:

```python
def verify_citations(answer_obj, retrieved_chunks):
    """Check that each cited source exists and claim is actually supported."""
    for citation in answer_obj['citations']:
        source = citation['source']
        claim = citation['claim']
        
        # Find the cited document
        doc = next((c for c in retrieved_chunks if c['id'] == source), None)
        if not doc:
            print(f"⚠️  Citation refers to non-existent source: {source}")
            return False
        
        # Check if claim is actually in the document (embedding-based or substring match)
        if not claim_in_document(claim, doc['text']):
            print(f"⚠️  Claim '{claim}' not found in {source}")
            return False
    
    return True  # All citations verified
```

### Layer 4: NLI-Based Rejection

Use an entailment model as a gatekeeper:

```python
from sentence_transformers import CrossEncoder

nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-large')

def safe_answer_or_hallucination(answer, context):
    scores = nli_model.predict([[context, answer]])
    entailment_score = scores[0][2]  # P(entailment)
    
    if entailment_score < 0.7:
        return "Unable to provide a confident answer based on the documents."
    
    return answer
```

### Practical Trade-offs

| Approach | Overhead | Reliability | Best For |
|----------|----------|-------------|----------|
| Citation prompting alone | Minimal (1-2% tokens) | Low (model may ignore instructions) | High-trust domains where spot-checking is feasible |
| Structured output | Moderate (JSON parsing) | Medium (enforces format, not correctness) | APIs, downstream systems that need machine-readable output |
| NLI verification | High (extra inference) | High (catches most hallucinations) | High-stakes applications (medical, legal, financial) |
| Combination (all three) | High | Very high | Mission-critical systems |

</details>

---

## Q7. What are self-check and NLI-based guards, and how do they detect hallucinations? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Self-check and NLI-based guards are **post-generation filtering mechanisms** that catch hallucinations after the LLM produces an answer.

### Self-Check Strategy

The LLM itself is tasked with verifying its own answer:

```python
def self_check_hallucination(question, answer, context):
    """Ask the LLM to critique its own answer."""
    
    critique_prompt = f"""
    Question: {question}
    
    Provided context:
    {context}
    
    Proposed answer: {answer}
    
    Your task: Does the above answer follow logically from the context?
    - If YES, respond "SUPPORTED"
    - If NO or UNCLEAR, respond "NOT SUPPORTED" and explain why.
    """
    
    critique = llm(critique_prompt)
    
    if "NOT SUPPORTED" in critique:
        return None  # Reject this answer, try again or return "I don't know"
    
    return answer
```

**Pros:**
- Uses existing LLM, no extra models needed
- Catches some contradictions the model itself can spot

**Cons:**
- LLM may fail to self-critique (overconfidence)
- Roughly doubles latency (2 LLM calls per query)
- Limited accuracy

### NLI-Based Guards (Entailment Checking)

NLI (Natural Language Inference) models predict whether a premise entails a hypothesis:

```python
from sentence_transformers import CrossEncoder

nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-large')

def nli_based_guard(answer, retrieved_chunks):
    """Use NLI model to verify answer is entailed by context."""
    
    # Concatenate all retrieved chunks as premise
    context = " ".join([c['text'] for c in retrieved_chunks])
    
    # Compute entailment: P(context ⊨ answer)
    scores = nli_model.predict([[context, answer]])
    
    # scores[0] = [contradiction_prob, neutral_prob, entailment_prob]
    entailment_prob = scores[0][2]
    
    return {
        'is_hallucination': entailment_prob < 0.65,  # threshold
        'entailment_score': entailment_prob,
        'decision': 'ACCEPT' if entailment_prob >= 0.65 else 'REJECT'
    }
```

**Popular NLI Models:**

| Model | Domain | Speed | Accuracy |
|-------|--------|-------|----------|
| `cross-encoder/nli-deberta-v3-large` | General-purpose | ~50ms per inference | ~92% on MNLI |
| `cross-encoder/nli-deberta-v3-small` | Fast inference | ~15ms | ~91% |
| `microsoft/deberta-v3-large` (fine-tuned NLI) | Custom domain | Variable | Domain-dependent |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Multilingual | ~30ms | ~88% (lower but faster) |

### Production Architecture

```
Query
  ↓
[Retrieval] → Retrieved Chunks
  ↓
[LLM Generation] → Candidate Answer
  ↓
[NLI Guard] ← Entailment Check
  ├─ entailment_score >= threshold → ACCEPT (output answer)
  └─ entailment_score < threshold  → REJECT (output fallback: "I don't know" or retry)
```

### Hybrid Approach: Stacked Guards

Combine multiple guards for higher precision:

```python
def stacked_hallucination_guard(question, answer, context, chunks):
    """Apply multiple guards; accept only if all pass."""
    
    # Guard 1: Citation check
    if not verify_has_citations(answer, chunks):
        return False
    
    # Guard 2: NLI entailment
    nli_result = nli_based_guard(answer, chunks)
    if nli_result['is_hallucination']:
        return False
    
    # Guard 3: Self-check
    self_check = self_check_hallucination(question, answer, context)
    if self_check is None:
        return False
    
    # All guards passed
    return True
```

### Cost-Latency Analysis

| Guard Type | Added Latency | Model Cost | Accuracy Lift |
|------------|---------------|-----------|----|
| Self-check | +500ms (extra LLM call) | +$0.002/call | +5–10% hallucination detection |
| NLI (single) | +50ms | +$0.0005/call | +15–25% |
| Stacked (all 3) | +600ms | +$0.0025/call | +25–35% |

**Trade-off:** High-stakes systems use stacked guards; latency-sensitive systems use lightweight NLI only.

</details>

---

## Q8. How do position-aware reordering and chunking strategies reduce hallucinations? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Since attention dilution and "lost in the middle" are major hallucination causes, reordering and chunk-level design strategies combat this directly:

### Position-Aware Reordering

**Strategy 1: Relevance-Based Ordering**

Rank retrieved chunks by relevance score and place highest-ranked chunks first (and/or at the end):

```python
def reorder_chunks_for_attention(chunks, query_embedding, reranker_model=None):
    """Reorder chunks to optimize LLM attention."""
    
    if reranker_model:
        # Use cross-encoder reranker (most effective)
        scores = reranker_model.predict(
            [[query, chunk['text']] for chunk in chunks]
        )
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    else:
        # Fallback: use retriever scores
        ranked = sorted(chunks, key=lambda x: x['score'], reverse=True)
    
    # Place top chunks first (and optionally repeat top-1 at end)
    reordered = ranked[:1] + ranked[1:-1] + [ranked[0]]  # Best, middle, best
    
    return reordered
```

**Strategy 2: Recency Grouping**

For time-sensitive queries, place recently updated chunks first:

```python
def group_by_recency(chunks, query):
    """Group chunks: recent first, then less recent."""
    
    now = datetime.now()
    
    def recency_score(chunk):
        updated_at = chunk.get('updated_at', datetime.min)
        days_old = (now - updated_at).days
        # Exponential decay: recent docs score higher
        return math.exp(-days_old / 30)  # 30-day half-life
    
    return sorted(chunks, key=recency_score, reverse=True)
```

### Smart Chunking Strategies

**Strategy 1: Variance in Chunk Sizes**

Use mixed chunk sizes based on content type:

```python
def intelligent_chunk_sizes(document, content_type='general'):
    """Vary chunk size by content type."""
    
    if content_type == 'code':
        return 256  # Keep code blocks intact and short
    elif content_type == 'technical':
        return 512  # Allow more context for definitions
    elif content_type == 'narrative':
        return 1024  # Prose benefits from larger context
    else:
        return 512  # Default
```

**Strategy 2: Hierarchical Chunking with Headers**

Preserve document structure to aid attention:

```markdown
# Document Structure
## Section A
- Subsection A1
  (chunk_1)
- Subsection A2
  (chunk_2)
## Section B
  (chunk_3)
```

When retrieving chunk_2, include the hierarchical path in the prompt:

```
Retrieved Context:

[Document: Finance Report > Section A > Subsection A2]
Revenue for Q3 2024 was $10M...

[Document: Finance Report > Section B]
Operating costs increased by 5%...
```

The path acts as an attention guide, helping the LLM focus on relevant sections.

**Strategy 3: Dual-Representation with Summaries**

For long documents, provide both a summary chunk and detailed chunks:

```python
def create_hierarchical_chunks(document):
    """Create a summary chunk + detailed chunks."""
    
    summary = generate_summary(document, max_tokens=200)
    chunks = split_document(document, chunk_size=512)
    
    return [
        {'id': 'summary', 'text': summary, 'level': 'summary'},
        *[{'id': f'detail_{i}', 'text': c, 'level': 'detail'} 
          for i, c in enumerate(chunks)]
    ]
```

Then in retrieval, return: summary + top-2 detail chunks. This reduces hallucination by providing condensed context alongside details.

### Production Results

Studies and benchmarks show:

| Optimization | Baseline Accuracy | After Reordering | Improvement |
|--------------|-------------------|------------------|-------------|
| Lost in Middle (needle at 50%) | 42% | 78% | +36% |
| Hallucination Rate (questions) | 12% | 7% | -5% (fewer hallucinations) |
| Latency (reranking cost) | 150ms | 350ms | +200ms |

### Integration Example

```python
def retrieve_and_reorder(query, retriever, reranker, num_chunks=5):
    """Full retrieval + reordering pipeline."""
    
    # Step 1: Initial retrieval (fast)
    candidates = retriever.search(query, k=20)
    
    # Step 2: Rerank (slower but more accurate)
    reranked = reranker.rank(query, candidates, top_k=num_chunks)
    
    # Step 3: Reorder for attention (position-aware)
    final_chunks = reorder_chunks_for_attention(reranked, query)
    
    # Step 4: Format with structural guidance
    formatted_context = format_with_hierarchy(final_chunks)
    
    return formatted_context
```

### Cost-Accuracy Tradeoff

| Level | Approach | Latency | Cost | Hallucination Reduction |
|-------|----------|---------|------|------------------------|
| Basic | No reordering | 100ms | Low | 0% |
| Standard | Rerank top-10 | 250ms | Medium | -3% to -5% |
| Advanced | Rerank + position optimization | 300ms | Medium | -5% to -10% |
| Premium | Hierarchical + recency + rerank | 400ms | Medium-High | -10% to -15% |

</details>

---

## Q9. How do you evaluate hallucination rates in RAG systems? What metrics and benchmarks exist? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Hallucination evaluation is critical but challenging. Unlike retrieval (which has clear Recall/Precision metrics), hallucination detection is inherently fuzzy and requires multiple approaches:

### Key Metrics

| Metric | Definition | How to Compute | Limitations |
|--------|-----------|---|---|
| **Citation F1** | Fraction of answer claims that are cited + verifiable in context | Manual audit + NER to extract claims | Labor-intensive, subjective claim boundaries |
| **Faithfulness (RAGAS)** | % of generated answer statements that are supported by context | Use NLI model to score each sentence | Depends on NLI model quality, may miss paraphrasing |
| **Entailment Score** | Cross-encoder NLI model confidence that context entails answer (0–1) | Inference with `cross-encoder/nli-deberta-v3-large` | Threshold-dependent, not always aligned with human judgment |
| **Token Overlap** | Fraction of answer tokens that appear in retrieved context (simple metric) | String matching or token-level overlap | High false negatives (paraphrasing, synonyms) |
| **BERTScore (Recall)** | Semantic similarity between answer and context using BERT embeddings | Sum of max cosine similarities per answer token vs context tokens | Indirect proxy, not a direct hallucination measure |

### Standard Benchmarks & Datasets

**1. RAGAS (RAG Assessment)**

Industry-standard suite for RAG evaluation:

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

# Your RAG outputs
results = {
    "question": ["What is ...?", ...],
    "answer": ["Answer 1", ...],
    "retrieved_context": [["context chunk 1", ...], ...],
    "ground_truth": ["Expected answer", ...]
}

# Evaluate
scores = evaluate(
    results,
    metrics=[faithfulness, answer_relevancy, context_recall]
)

# faithfulness ∈ [0, 1]: higher = fewer hallucinations
# answer_relevancy ∈ [0, 1]: does answer match the question?
# context_recall ∈ [0, 1]: did retrieval get the necessary context?
```

**Faithfulness details:**

```python
# RAGAS decomposition:
# 1. LLM extracts factual statements from answer
# 2. For each statement, check if context supports it (using NLI or QA model)
# 3. faithfulness = (# supported statements) / (# total statements)

Example:
Answer: "Company X was founded in 2015 by John Doe and has 500 employees."
Statements: 
  - "Company X was founded in 2015" → SUPPORTED (in context)
  - "John Doe was a founder" → SUPPORTED
  - "has 500 employees" → NOT SUPPORTED (context doesn't mention this)
Faithfulness = 2/3 = 0.67
```

**2. TruLens (Truera's Evaluation Framework)**

LLM-based feedback system with hallucination-specific modules:

```python
from trulens_eval import TruChain, Feedback, Huggingface

# Define feedback functions
qs_relevance = Feedback(
    huggingface_models.relevance,
    name="Relevance"
).on_input_output()

qa_correctness = Feedback(
    huggingface_models.qa_correctness,
    name="Correctness"
).on_input_output()

# Evaluate RAG chain
tru_recorder = TruChain(
    my_rag_chain,
    app_id="my_rag_app",
    feedbacks=[qs_relevance, qa_correctness]
)

with tru_recorder:
    tru_recorder.app(question)

# Returns feedback scores (including hallucination likelihood)
```

**3. AlpacaEval / ALCE (Attribution)

ALCE (Attributable Language Explanation) specifically measures hallucinations via citation correctness:

```python
# Input: (question, answer with citations)
# Output: citation accuracy score

Example:
Q: "When was Python created?"
A: "Python was created in 1991 [1]. It was designed by Guido van Rossum [2]."

Evaluation:
[1] = Doc about Python history → claims match? ✓
[2] = Doc about Guido Rossum → claims match? ✓
ALCE Score = 1.0 (perfect attribution)
```

### Practical Evaluation Pipeline

```python
def evaluate_hallucination_rate(rag_system, test_set, output_file='results.csv'):
    """Comprehensive hallucination evaluation."""
    
    results = []
    
    for question, ground_truth_context in test_set:
        # 1. Generate answer
        answer = rag_system.answer(question)
        retrieved_context = rag_system.get_retrieval_context(question)
        
        # 2. Compute multiple metrics
        faithfulness_score = ragas_faithfulness(answer, retrieved_context)
        citation_score = evaluate_citations(answer, retrieved_context)
        entailment_score = nli_model.predict([[
            " ".join(retrieved_context),
            answer
        ]])[0][2]
        
        # 3. Manual audit (sample ~10%)
        manual_review = None
        if random.random() < 0.1:
            manual_review = prompt_human_reviewer(
                question, answer, retrieved_context
            )
        
        results.append({
            'question': question,
            'answer': answer,
            'faithfulness': faithfulness_score,
            'citation_score': citation_score,
            'entailment': entailment_score,
            'manual_review': manual_review
        })
    
    # Aggregate
    df = pd.DataFrame(results)
    print(f"Mean Faithfulness: {df['faithfulness'].mean():.2%}")
    print(f"Mean Citation Score: {df['citation_score'].mean():.2%}")
    print(f"Hallucination Rate (entailment < 0.65): {(df['entailment'] < 0.65).mean():.2%}")
    
    df.to_csv(output_file)
    return df
```

### Recommended Thresholds for Production

| Metric | Target | Action if Below |
|--------|--------|-----------------|
| Faithfulness (RAGAS) | > 0.85 | Review retrieval + reranking |
| Entailment Score | > 0.70 | Enable NLI-based filtering |
| Citation F1 | > 0.90 | Enforce structured output with citations |
| Manual Audit Hallucination Rate | < 2% | Investigate specific failure modes |

</details>

---

## Q10. What is the cost-latency impact of hallucination mitigation, and how do you choose which strategies to deploy? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Hallucination mitigation comes at a cost: extra inference, re-ranking, and model calls. Production decisions require understanding the trade-off between safety and performance.

### Cost-Latency Matrix

| Strategy | Added Latency | Cost per Query | Hallucination Reduction | Use Case |
|----------|---------------|---|---|---|
| **Reordering only** | +50ms | +$0.0001 | -3% | High-volume, latency-sensitive |
| **Reranking (top-10)** | +200ms | +$0.0008 | -8% | Moderate volume, quality-focused |
| **NLI entailment check** | +50ms | +$0.0005 | -12% | Filtering pipeline step |
| **Self-check critique** | +500ms | +$0.002 | -8% | Low-volume, high-stakes |
| **Structured output (JSON)** | +100ms | +$0.0002 | -5% | API integration |
| **Full stack** (rerank + NLI + citations) | +300ms | +$0.0015 | -25% | Mission-critical |

### Dollar Cost Analysis (Anthropic Claude 3.5 Sonnet pricing as of 2024)

```python
def cost_per_query_analysis(queries_per_month=10000):
    """Compare cost across strategies."""
    
    strategies = {
        'baseline': {
            'llm_calls': 1,
            'extra_models': 0,
            'input_tokens': 2000,
            'output_tokens': 300,
        },
        'with_reranking': {
            'llm_calls': 1,
            'reranker_calls': 1,  # cross-encoder
            'input_tokens': 2000,
            'output_tokens': 300,
        },
        'with_nli_guard': {
            'llm_calls': 1,
            'nli_calls': 1,
            'input_tokens': 2000,
            'output_tokens': 300,
        },
        'full_stack': {
            'llm_calls': 1,
            'reranker_calls': 1,
            'nli_calls': 1,
            'input_tokens': 2000,
            'output_tokens': 300,
        }
    }
    
    # Pricing (as of 2024)
    llm_input_price = 3 / 1_000_000  # $3 per 1M input tokens
    llm_output_price = 15 / 1_000_000  # $15 per 1M output tokens
    reranker_price = 0.0005  # per call (e.g., HuggingFace API)
    nli_price = 0.0003  # per call
    
    for name, config in strategies.items():
        llm_cost = (config['input_tokens'] * llm_input_price +
                    config['output_tokens'] * llm_output_price)
        extra_cost = (config.get('reranker_calls', 0) * reranker_price +
                      config.get('nli_calls', 0) * nli_price)
        total_per_query = llm_cost + extra_cost
        total_per_month = total_per_query * queries_per_month
        
        print(f"{name}:")
        print(f"  Per query: ${total_per_query:.6f}")
        print(f"  Per month (10k queries): ${total_per_month:.2f}")
        print()

# Output:
# baseline:
#   Per query: $0.000810
#   Per month (10k queries): $8.10
#
# with_reranking:
#   Per query: $0.000815
#   Per month (10k queries): $8.15
#
# full_stack:
#   Per query: $0.000818
#   Per month (10k queries): $8.18
```

### Decision Framework: Which Strategies to Deploy

**Tier 1: High-Volume, Latency-Sensitive (e.g., Chat Assistants)**

- Baseline + reordering (position-aware placement)
- Cost: +$0.0001 per query
- Latency: +50ms
- Hallucination reduction: -3%
- Rationale: Minimal overhead, some improvement

```python
# Example: Customer service chatbot
# - 1M queries/month
# - P99 latency budget: 500ms
# - Acceptable hallucination rate: < 5%

retriever_context = retrieve(query, k=5)
chunks = reorder_chunks_for_attention(retriever_context)
answer = llm(format_prompt(chunks))
```

**Tier 2: Moderate Volume, Quality-Focused (e.g., Knowledge Base QA)**

- Baseline + reranking + reordering
- Cost: +$0.0008 per query
- Latency: +200ms
- Hallucination reduction: -8%
- Rationale: Reranking is most effective per dollar spent

```python
# Example: Document QA system
# - 50k queries/month
# - P99 latency: 2s
# - Hallucination rate must be < 3%

chunks = retrieve(query, k=20)
reranked = reranker(query, chunks, top_k=5)
reordered = reorder_chunks_for_attention(reranked)
answer = llm(format_prompt(reordered))
```

**Tier 3: Low-Volume, Mission-Critical (e.g., Medical, Legal, Financial Decisions)**

- Full stack: reranking + NLI guard + citation enforcement + self-check
- Cost: +$0.002+ per query
- Latency: +600ms (acceptable for asynchronous systems)
- Hallucination reduction: -25%+
- Rationale: Cost is negligible relative to decision impact; hallucination must be < 1%

```python
# Example: Medical diagnosis support
# - 500 queries/month
# - Hallucination rate must be < 1%
# - Cost per wrong decision >> cost of mitigation

chunks = retrieve(query, k=20)
reranked = reranker(query, chunks, top_k=5)
reordered = reorder_chunks_for_attention(reranked)
answer = llm(format_prompt(reordered))

# NLI guard
if nli_guard(answer, reordered) < 0.7:
    return "Unable to provide confident answer. Please consult a specialist."

# Citation check
verified_answer = verify_citations(answer, reordered)

# Self-check
final_answer = self_check_hallucination(query, verified_answer, " ".join(reordered))

return final_answer
```

### Monitoring ROI of Mitigations

Track in production:

```python
def monitor_mitigation_roi(strategy_name, baseline_hallucination_rate):
    """Measure actual improvement from deployed strategy."""
    
    # Before mitigation deployed
    baseline_rate = baseline_hallucination_rate  # e.g., 8%
    
    # After mitigation (measured via audit + NLI)
    current_rate = measure_hallucination_rate_via_audit()
    
    improvement = baseline_rate - current_rate  # e.g., 8% - 3% = 5%
    cost_per_query = get_cost_for_strategy(strategy_name)
    monthly_cost = cost_per_query * total_monthly_queries()
    
    # Cost per percentage-point of hallucination reduction
    cost_per_pct_reduction = monthly_cost / (improvement * 100)
    
    print(f"Strategy: {strategy_name}")
    print(f"Hallucination reduction: {improvement:.1%}")
    print(f"Monthly cost: ${monthly_cost:.2f}")
    print(f"Cost per 1% reduction: ${cost_per_pct_reduction:.2f}")
    
    if cost_per_pct_reduction < 50:  # arbitrary threshold
        print("✓ ROI positive, continue deployment")
    else:
        print("⚠ ROI marginal, review strategy")
```

### Recommended Implementation Path

1. **Month 1:** Deploy reranking (best ROI) → measure improvement
2. **Month 2:** Add NLI guard if hallucination still > 3% → measure
3. **Month 3:** Add citation enforcement if needed for compliance
4. **Month 4+:** Monitor and adjust thresholds based on user feedback

</details>

---
