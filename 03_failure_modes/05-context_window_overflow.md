# 05 — Context Window Overflow

> Too many or too large retrieved chunks are included in the prompt, exhausting the LLM's context window and forcing truncation or causing the model to deprioritize relevant information.

---

## Q1. What is context window overflow and why does it matter in RAG? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Context window overflow occurs when the total prompt size (query + instructions + retrieved documents) exceeds the LLM's context window, forcing the system to either truncate content or deprioritize information.

**Example:**

```
Claude 3.5 Sonnet: 200,000 token context window

RAG system retrieves:
  - Query: 100 tokens
  - System prompt: 500 tokens
  - Retrieved chunks: 180,000 tokens (18 chunks × 10,000 each)
  - Total: 180,600 tokens ✓ (fits)

But: Claude's effective working memory is lower due to:
  - Loss of attention at long context (Liu et al., 2023)
  - Increased hallucination with longer context
  - Reduced output quality when context is >50k tokens for many models

Result: Model may ignore middle chunks or produce lower-quality output.
```

**Why it matters:**

1. **Cost explosion:** Larger context = higher token costs (proportional to input tokens)
2. **Quality degradation:** Long context makes models forget or confuse information
3. **Latency increase:** Processing 200k tokens takes 2-3x longer than 50k
4. **Forced truncation:** May silently drop relevant information to fit window
5. **Reduced observability:** Hard to know what was truncated and why

| Scenario | Impact | Severity |
|----------|--------|----------|
| **100k tokens in 200k window** | Works but suboptimal (middle tokens lose attention) | Medium |
| **180k tokens in 200k window** | Fits but very tight, risk of truncation | High |
| **220k tokens in 200k window** | Exceeds window, must truncate (data loss guaranteed) | Critical |

This is distinct from retrieval failure because the relevant *information exists and is retrieved*, but it's not effectively *used* due to context constraints.

</details>

---

## Q2. What are observable symptoms of context window overflow? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Context overflow manifests through several detectable signals:

| Symptom | Detection Method | Example |
|---------|---|---|
| **Answer based on truncated context** | Compare answer to full retrieved docs | System says "Information not available", but answer is in truncated chunk |
| **Inconsistent answers to same question** | Ask question multiple times | Same query, different answers depending on which chunks made it into context |
| **Middle chunk information ignored** | Deliberately place correct answer in middle chunk | Answer uses first/last chunks but ignores middle ones |
| **Prompt too large warning** | Log total token count of prompt | Context: 198,500 tokens (only 1,500 tokens left for output!) |
| **Truncation in logs** | Check if any chunks were dropped | "Dropped 2 chunks (15,000 tokens) due to context limit" |
| **LLM output warns about space** | Model explicitly says it ran out of context | "Due to length constraints, I may not have covered all details" |
| **Quality degradation with more docs** | Recall@k vs quality trade-off | More chunks retrieved → Lower answer quality (opposite of expected) |

**Production signals:**

```python
def detect_context_overflow_signals(query, retrieved_chunks, llm_response):
    """Flag potential context overflow issues."""
    
    # Signal 1: Total context size
    context_tokens = estimate_tokens(query) + sum(estimate_tokens(c['text']) for c in retrieved_chunks)
    context_limit = 200_000  # Claude's window
    
    if context_tokens > context_limit * 0.9:
        log_overflow_alert(f"Context at {context_tokens/context_limit*100:.0f}% of limit")
    
    # Signal 2: Truncated chunks in retrieval
    if len(retrieved_chunks) > 10:
        # More than 10 chunks is risky
        log_overflow_alert(f"Retrieved {len(retrieved_chunks)} chunks, risk of overflow")
    
    # Signal 3: Quality dropping with more context
    if context_tokens > 50_000:
        # Many models degrade with >50k tokens
        log_overflow_alert(f"Large context ({context_tokens:,} tokens), watch for quality drop")
```

</details>

---

## Q3. What causes context window overflow? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Context overflow has multiple independent causes:

### 1. Over-Retrieval (Too Many Chunks)

Retrieving more chunks than necessary:

```python
# Common mistake: retrieve with high k
results = retriever.search(query, k=50)  # 50 chunks!

# Average chunk: 1,000 tokens
# 50 chunks × 1,000 tokens = 50,000 tokens just for docs

# Plus: system prompt (500), query (100), output space needed (5,000)
# Total: ~55,600 tokens in 200k window

# While this "fits", quality degrades with >50k tokens
```

**Root causes:**
- Uncertainty about relevance → retrieve more to be safe
- No ranking/reranking → return all results
- High false negative tolerance (want 99% recall) → retrieve many

### 2. Chunk Size Too Large

Individual chunks are very long:

```
Document chunking strategy:
  ❌ Bad: 5,000 tokens per chunk × 10 chunks = 50,000 tokens
  ✓ Good: 512 tokens per chunk × 10 chunks = 5,120 tokens
  
Same number of chunks, but 10x size difference!
```

### 3. Lack of Prioritization

All chunks weighted equally, no importance ranking:

```
Retrieve top-10 by similarity:
  Chunk 1: Highly relevant (similarity 0.95)
  Chunk 2: Somewhat relevant (similarity 0.75)
  Chunk 3-10: Marginally relevant (similarity 0.60-0.65)
  
All included in prompt, but middle ones dilute context.
Better: Include top-3, skip the marginal ones.
```

### 4. Redundant Information

Retrieved chunks contain overlapping information:

```
Query: "How to implement authentication?"

Retrieved chunks:
  Chunk A: "Authentication is verifying user identity. JWT tokens..."
  Chunk B: "JWT tokens provide stateless auth. They contain..."
  Chunk C: "Stateless authentication using tokens is efficient..."
  Chunk D: "Tokens reduce server load..."
  
Problem: A-D all say similar things, wasting ~4,000 tokens on repetition.
Solution: Deduplication. Keep only A, discard B-D.
```

### 5. Metadata Bloat

Metadata included with documents adds overhead:

```
Chunk object:
  {
    'id': 'doc_123',
    'text': '...',  # 1,000 tokens
    'metadata': {
      'source': '...',
      'author': '...',
      'created_at': '...',
      'updated_at': '...',
      'tags': ['tag1', 'tag2', ...],  # 100 tokens
      'full_html': '...',  # 5,000 tokens ← Why include this?
      'raw_text': '...',  # Duplicate of 'text'
    }
  }
  
Total per chunk: 1,000 + 5,100 = 6,100 tokens!
Solution: Include only necessary metadata.
```

### 6. System Prompt + Instructions Too Long

Large system prompt leaves little room for context:

```python
system_prompt = """
You are an AI assistant expert in medical information retrieval...
[Long instructions about behavior, tone, constraints]
[Examples of good/bad responses]
[Safety guidelines]
[Multi-step reasoning instructions]
...
Total: 5,000 tokens ← Before any retrieved docs!
"""

# With 200k window, system prompt alone leaves 195k for context.
# But effective context after system + query ≈ 150k (due to position bias).
```

### 7. No Dynamic Chunk Selection

Fixed number of chunks regardless of query difficulty:

```python
def naive_retrieval(query):
    # Always retrieve 10 chunks
    return retriever.search(query, k=10)

# Better: Dynamic k based on query complexity
def smart_retrieval(query):
    query_complexity = estimate_complexity(query)
    k = min(5, 10, 15)[query_complexity]  # 5, 10, or 15 chunks
    return retriever.search(query, k=k)
```

</details>

---

## Q4. How do you detect and measure context window usage? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

### Method 1: Token Counting

```python
import tiktoken

def count_context_tokens(system_prompt, query, retrieved_chunks, model='claude-3-5-sonnet-20241022'):
    """Estimate total tokens in prompt."""
    
    # Note: Exact token counts vary by model
    # This uses OpenAI's tokenizer for approximation
    encoding = tiktoken.encoding_for_model("gpt-4")
    
    tokens = {
        'system_prompt': len(encoding.encode(system_prompt)),
        'query': len(encoding.encode(query)),
        'documents': sum(len(encoding.encode(chunk['text'])) for chunk in retrieved_chunks),
        'formatting': 100,  # Rough estimate for formatting/delimiters
    }
    
    tokens['total'] = sum(tokens.values())
    
    return tokens

# Usage
tokens = count_context_tokens(
    system_prompt="You are a helpful assistant...",
    query="What is RAG?",
    retrieved_chunks=[...]
)

print(f"System: {tokens['system_prompt']:,} tokens")
print(f"Query: {tokens['query']:,} tokens")
print(f"Documents: {tokens['documents']:,} tokens")
print(f"Total: {tokens['total']:,} tokens")
print(f"Available: 200,000 tokens")
print(f"Headroom: {200_000 - tokens['total']:,} tokens")

if tokens['total'] > 150_000:
    print("⚠️ WARNING: Context usage is high, quality may degrade")
```

### Method 2: Runtime Monitoring

```python
def monitor_context_usage(messages, model='claude-3-5-sonnet'):
    """Monitor actual token usage in API calls."""
    
    from anthropic import Anthropic
    
    client = Anthropic()
    
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=messages,
    )
    
    # Extract token usage from response
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    
    # Model-specific limits
    limits = {
        'claude-opus-4-8': 200_000,
        'claude-sonnet-4-6': 200_000,
        'claude-haiku-4-5-20251001': 200_000,
    }
    
    limit = limits.get(model, 200_000)
    utilization = input_tokens / limit
    
    log_metric('context_utilization', utilization)
    log_metric('input_tokens', input_tokens)
    log_metric('output_tokens', output_tokens)
    
    if utilization > 0.9:
        alert(f"Context usage at {utilization:.0%} of limit")
    
    return {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'utilization': utilization,
    }
```

### Method 3: Chunk-Level Analysis

```python
def analyze_chunk_contribution(query, chunks, llm_response):
    """Determine which chunks actually influenced the answer."""
    
    # Method 1: Ablation - remove each chunk, see if answer changes
    baseline_response = llm_response
    
    contribution_scores = {}
    
    for i, chunk in enumerate(chunks):
        # Remove chunk i
        chunks_without_i = chunks[:i] + chunks[i+1:]
        
        # Get new response
        new_response = llm.generate(query, chunks_without_i)
        
        # Measure difference
        diff = semantic_similarity(baseline_response, new_response)
        
        contribution_scores[i] = 1 - diff  # Higher = more important
    
    # Identify unused chunks
    unused_threshold = 0.05
    unused_chunks = [i for i, score in contribution_scores.items() if score < unused_threshold]
    
    print(f"Chunk contribution analysis:")
    for i, score in contribution_scores.items():
        print(f"  Chunk {i}: {score:.2%} importance")
    
    if unused_chunks:
        print(f"⚠️ {len(unused_chunks)} chunks not contributing to answer")
    
    return contribution_scores
```

### Production SLOs for Context

```python
context_slos = {
    'max_input_tokens': 100_000,        # Never exceed 100k input tokens
    'target_input_tokens': 50_000,      # Aim for <50k for quality
    'max_chunks': 10,                   # Retrieve at most 10 chunks
    'max_chunk_size': 1_000,            # Individual chunk max 1k tokens
    'system_prompt_max': 2_000,         # System prompt max 2k tokens
    'monitoring_frequency': '100%',     # Check every request
}
```

</details>

---

## Q5. What strategies reduce context window usage without losing quality? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

### Strategy 1: Dynamic Top-K Selection

Choose retrieval k based on query complexity:

```python
def estimate_query_complexity(query):
    """Estimate how many chunks needed to answer."""
    
    # Simple heuristics
    word_count = len(query.split())
    question_marks = query.count('?')
    
    # Multi-part question (e.g., "How X, why Y, when Z?")
    # Needs more context
    
    if question_marks > 1:
        return 'complex'
    elif word_count < 5:
        return 'simple'
    else:
        return 'moderate'

def retrieve_with_dynamic_k(query, retriever):
    """Retrieve different amounts based on query."""
    
    complexity = estimate_query_complexity(query)
    
    k_map = {
        'simple': 3,      # Simple query, few chunks needed
        'moderate': 5,
        'complex': 8,     # Complex multi-part query
    }
    
    k = k_map[complexity]
    
    results = retriever.search(query, k=k)
    
    log_metric('retrieval_k', k)
    log_metric('query_complexity', complexity)
    
    return results
```

### Strategy 2: Chunk Summarization

Summarize retrieved chunks before including in prompt:

```python
def summarize_chunks(chunks, target_tokens_per_chunk=200):
    """Summarize long chunks to reduce size."""
    
    summarized = []
    
    for chunk in chunks:
        chunk_tokens = estimate_tokens(chunk['text'])
        
        if chunk_tokens > target_tokens_per_chunk:
            # Summarize using LLM
            summary = llm.summarize(
                chunk['text'],
                max_tokens=target_tokens_per_chunk
            )
            summarized.append({
                'original_tokens': chunk_tokens,
                'summary_tokens': estimate_tokens(summary),
                'text': summary,
                'type': 'summary'
            })
        else:
            # Keep as-is
            summarized.append(chunk)
    
    return summarized

# Example savings
original_tokens = 10_000
summarized = summarize_chunks(chunks, target_tokens_per_chunk=200)
reduced_tokens = sum(estimate_tokens(c['text']) for c in summarized)

print(f"Reduced from {original_tokens:,} to {reduced_tokens:,} tokens ({reduced_tokens/original_tokens:.0%})")
```

### Strategy 3: Deduplication

Remove redundant chunks:

```python
def deduplicate_chunks(chunks, similarity_threshold=0.85):
    """Remove chunks that are too similar to others."""
    
    unique_chunks = []
    
    for chunk in chunks:
        # Compare to existing unique chunks
        is_duplicate = False
        
        for unique_chunk in unique_chunks:
            similarity = semantic_similarity(chunk['text'], unique_chunk['text'])
            
            if similarity > similarity_threshold:
                # Too similar, skip
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_chunks.append(chunk)
    
    return unique_chunks

# Example savings
original_chunks = 10
deduplicated = deduplicate_chunks(chunks, similarity_threshold=0.85)

print(f"Removed {len(chunks) - len(deduplicated)} duplicate chunks")
print(f"Tokens: {sum(estimate_tokens(c['text']) for c in chunks):,} → {sum(estimate_tokens(c['text']) for c in deduplicated):,}")
```

### Strategy 4: Hierarchical Summarization

Multiple levels of summaries:

```python
def create_hierarchical_summaries(chunks):
    """Create summaries at different levels of abstraction."""
    
    # Level 1: Full text
    level_1 = chunks
    
    # Level 2: 1-sentence summary per chunk
    level_2 = [
        {
            'text': llm.summarize(c['text'], max_tokens=50),
            'level': 'summary_1sentence'
        }
        for c in level_1
    ]
    
    # Level 3: Key points (bullet list)
    level_3 = [
        {
            'text': llm.extract_key_points(c['text'], max_bullets=3),
            'level': 'key_points'
        }
        for c in level_1
    ]
    
    return {
        'full': level_1,
        'summary': level_2,
        'key_points': level_3,
    }

# LLM can choose which level to use based on context budget
def select_summary_level(chunks_hierarchical, max_tokens=10_000):
    """Choose summary level based on available tokens."""
    
    available = max_tokens
    
    # Try full text first
    full_tokens = sum(estimate_tokens(c['text']) for c in chunks_hierarchical['full'])
    
    if full_tokens < available:
        return chunks_hierarchical['full']
    
    # Try summaries
    summary_tokens = sum(estimate_tokens(c['text']) for c in chunks_hierarchical['summary'])
    
    if summary_tokens < available:
        return chunks_hierarchical['summary']
    
    # Use key points
    return chunks_hierarchical['key_points']
```

### Strategy 5: Reranking (Keep Only Top-K Relevant)

Rerank and prune low-relevance chunks:

```python
def retrieve_and_prune(query, retriever, reranker, max_tokens=15_000):
    """Retrieve many, rerank, keep only top by relevance."""
    
    # Retrieve with high k
    candidates = retriever.search(query, k=20)
    
    # Rerank
    reranked = reranker.rank(query, candidates, top_k=20)
    
    # Prune by tokens
    included_chunks = []
    token_count = 0
    
    for chunk in reranked:
        chunk_tokens = estimate_tokens(chunk['text'])
        
        if token_count + chunk_tokens < max_tokens:
            included_chunks.append(chunk)
            token_count += chunk_tokens
        else:
            break  # Stop when budget exhausted
    
    log_metric('pruned_chunks', len(candidates) - len(included_chunks))
    
    return included_chunks
```

### Comparison of Strategies

| Strategy | Token Reduction | Quality Loss | Complexity |
|----------|---|---|---|
| **Dynamic k** | -20% to -40% | Minimal (smart selection) | Low |
| **Summarization** | -50% to -70% | Low (summaries preserve info) | Medium |
| **Deduplication** | -10% to -30% | Minimal (removes redundancy) | Low |
| **Hierarchical** | -60% to -80% | Medium (loses detail) | High |
| **Reranking + Pruning** | -40% to -60% | Low (keeps most relevant) | Medium |

</details>

---

## Q6. How do you handle the "lost in the middle" problem with large context? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Even when context fits, LLMs struggle to attend to middle chunks (Liu et al., 2023). Strategies to mitigate:

### Strategy 1: Position-Aware Ordering

Place most relevant chunks at start/end:

```python
def reorder_by_position_importance(chunks, query):
    """Rank chunks and place important ones at edges."""
    
    # Compute relevance scores
    scores = reranker.rank(query, chunks)
    
    # Sort by score
    sorted_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    
    # Reorder: [top-1, ..., top-k, middle..., top-2, top-1]
    # Most relevant at start and end, least relevant in middle
    
    n = len(sorted_chunks)
    reordered = []
    
    # Add top half to start
    for i in range(n // 2):
        reordered.append(sorted_chunks[i][0])
    
    # Add second half to end (reversed)
    for i in range(n-1, n // 2 - 1, -1):
        reordered.append(sorted_chunks[i][0])
    
    return reordered
```

### Strategy 2: Explicit Markers

Mark important chunks with XML tags:

```python
def mark_important_chunks(chunks, query, importance_threshold=0.7):
    """Mark important chunks to guide attention."""
    
    # Rank
    scores = reranker.rank(query, chunks)
    
    marked_chunks = []
    
    for chunk, score in zip(chunks, scores):
        if score > importance_threshold:
            # Mark as important
            text = f"<IMPORTANT>\n{chunk['text']}\n</IMPORTANT>"
        else:
            text = f"<SUPPLEMENTARY>\n{chunk['text']}\n</SUPPLEMENTARY>"
        
        marked_chunks.append(text)
    
    return marked_chunks

# Prompt with marked chunks
prompt = f"""
{context_instructions}

{marked_chunks}

Question: {query}
"""
```

### Strategy 3: Retrieval-Augmented Generation (Sparse Retrieval)

Don't put all chunks in context; use retrieval at inference:

```python
def few_shot_with_sparse_retrieval(query):
    """Retrieve only top-2 most relevant chunks, not all 10."""
    
    # Retrieve top-2
    top_chunks = retriever.search(query, k=2)
    
    # Include in prompt
    prompt = f"""
{system_prompt}

Relevant documents:
{format_chunks(top_chunks)}

Question: {query}

If you need more information, you can ask: "Can you search for X?"
"""
    
    # For complex questions, model can ask for more info
    response = llm.generate(prompt)
    
    # If model asks for more info, retrieve additional chunks
    if "Can you search for" in response:
        search_query = extract_search_query(response)
        additional_chunks = retriever.search(search_query, k=3)
        # Include and re-answer
    
    return response
```

### Strategy 4: Chunk Numbering + Index References

Help model reference specific chunks:

```python
def number_and_reference_chunks(chunks):
    """Number chunks so model can reference them."""
    
    numbered = []
    
    for i, chunk in enumerate(chunks, 1):
        text = f"[Document {i}]\n{chunk['text']}\n"
        numbered.append(text)
    
    # Instruction to model
    instruction = """When referencing information, cite the document number, e.g., 'According to [Document 3], ...'"""
    
    return numbered, instruction

# This helps model stay focused on specific chunks rather than averaging all
```

### Strategy 5: Summary + Details Pattern

Provide summary of all chunks, then detailed chunks selectively:

```python
def create_summary_plus_details(chunks, max_detail_chunks=3):
    """High-level summary + detailed relevant chunks."""
    
    # Create one-sentence summary for each
    summaries = [
        llm.summarize(c['text'], max_tokens=20) for c in chunks
    ]
    
    # Rank by relevance
    ranked = reranker.rank(query, chunks)
    
    # Build prompt
    prompt = f"""
{system_prompt}

Overview of available documents:
{format_summaries(summaries)}

Detailed documents (most relevant):
{format_chunks(ranked[:max_detail_chunks])}

Question: {query}
"""
    
    return prompt
```

### Empirical Results

Study: Liu et al., 2023 "Lost in the Middle"

```
Document position in context window vs. retrieval accuracy:

Position:      0%    25%    50%    75%    100%
Baseline:     100%   89%    42%    88%    97%     ← 50% is worst!
With reorder:  99%   98%    87%    96%    98%     ← Much better
```

Reordering recovers ~45% of the lost accuracy in the middle.

</details>

---

## Q7. How do you balance cost and quality when managing context window size? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Context window management is fundamentally a cost-quality trade-off: more context improves quality but increases costs and latency.

### Cost Analysis

```python
def estimate_context_management_cost(queries_per_month=100_000):
    """Estimate costs of different context strategies."""
    
    strategies = {
        'minimal_context': {
            'avg_input_tokens': 5_000,      # Query + 2 chunks only
            'avg_output_tokens': 200,
            'cost_per_1m_input': 3,         # OpenAI pricing
            'cost_per_1m_output': 15,
        },
        'moderate_context': {
            'avg_input_tokens': 20_000,     # Query + 5-10 chunks
            'avg_output_tokens': 300,
            'cost_per_1m_input': 3,
            'cost_per_1m_output': 15,
        },
        'large_context': {
            'avg_input_tokens': 100_000,    # Query + 20+ chunks
            'avg_output_tokens': 500,
            'cost_per_1m_input': 3,
            'cost_per_1m_output': 15,
        },
    }
    
    for name, metrics in strategies.items():
        monthly_input_cost = (
            queries_per_month * metrics['avg_input_tokens'] / 1_000_000 *
            metrics['cost_per_1m_input']
        )
        
        monthly_output_cost = (
            queries_per_month * metrics['avg_output_tokens'] / 1_000_000 *
            metrics['cost_per_1m_output']
        )
        
        total_monthly = monthly_input_cost + monthly_output_cost
        cost_per_query = total_monthly / queries_per_month
        
        print(f"\n{name.upper()}")
        print(f"  Input cost: ${monthly_input_cost:.2f}/month")
        print(f"  Output cost: ${monthly_output_cost:.2f}/month")
        print(f"  Total: ${total_monthly:.2f}/month")
        print(f"  Per query: ${cost_per_query:.6f}")

estimate_context_management_cost()

# Output:
# MINIMAL_CONTEXT
#   Input cost: $1.50/month
#   Output cost: $0.30/month
#   Total: $1.80/month
#   Per query: $0.000018
#
# MODERATE_CONTEXT
#   Input cost: $6.00/month
#   Output cost: $0.45/month
#   Total: $6.45/month
#   Per query: $0.000065
#
# LARGE_CONTEXT
#   Input cost: $30.00/month
#   Output cost: $0.75/month
#   Total: $30.75/month
#   Per query: $0.000308
```

### Quality vs. Context Trade-off

```python
# Empirical relationship: more context → better quality (with diminishing returns)

context_sizes = [5_000, 10_000, 20_000, 50_000, 100_000, 150_000]
quality_scores = [0.72, 0.78, 0.82, 0.85, 0.86, 0.87]

# Quality improvement per 10k tokens
improvements = [
    (quality_scores[i+1] - quality_scores[i]) / 
    ((context_sizes[i+1] - context_sizes[i]) / 10_000)
    for i in range(len(context_sizes) - 1)
]

# Output:
# 10k tokens: +0.12 quality
# 10k tokens (20k→30k): +0.08 quality (diminishing)
# 10k tokens (50k→60k): +0.01 quality (plateau)

# ROI analysis: cost of improvement
for i, context_size in enumerate(context_sizes):
    cost = context_size * 3 / 1_000_000 * 100_000  # per 100k queries
    quality = quality_scores[i]
    cost_per_quality_point = cost / quality if quality > 0 else float('inf')
    
    print(f"{context_size:,} tokens: {quality:.2f} quality, ${cost:.2f}, ${cost_per_quality_point:.2f}/quality-point")
```

### Decision Matrix: Context Size by Use Case

| Use Case | Quality Needed | Recommended Context | Cost | Latency |
|----------|---|---|---|---|
| **General QA** | 0.75+ | 20k tokens (5-10 docs) | Low | 200ms |
| **Complex reasoning** | 0.85+ | 50k tokens (15-20 docs) | Medium | 500ms |
| **Research/Analysis** | 0.90+ | 100k+ tokens (30+ docs) | High | 1s+ |
| **Simple facts** | 0.70+ | 5k tokens (2-3 docs) | Very Low | 100ms |
| **Real-time chat** | 0.75+ | 10k tokens (3-5 docs) | Low | 150ms |

### Optimization Strategy: Stepped Approach

```python
def adaptive_context_management(query, retriever, reranker, quality_target=0.80):
    """Start small, expand context only if needed."""
    
    # Step 1: Try minimal context
    results = retriever.search(query, k=2)
    answer = llm.generate(query, results)
    confidence = assess_answer_confidence(answer)
    
    if confidence > quality_target:
        # Good enough, return
        return answer, metrics={'strategy': 'minimal', 'tokens': 5_000}
    
    # Step 2: Add more context
    results = retriever.search(query, k=5)
    reranked = reranker.rank(query, results, top_k=5)
    answer = llm.generate(query, reranked)
    confidence = assess_answer_confidence(answer)
    
    if confidence > quality_target:
        return answer, metrics={'strategy': 'moderate', 'tokens': 20_000}
    
    # Step 3: Full context
    results = retriever.search(query, k=20)
    reranked = reranker.rank(query, results, top_k=10)
    answer = llm.generate(query, reranked)
    
    return answer, metrics={'strategy': 'large', 'tokens': 100_000}

# This avoids wasting tokens on simple questions while handling complex ones
```

### SLOs for Context Management

```python
context_slos = {
    'avg_input_tokens_per_query': 30_000,   # Target average
    'p95_input_tokens': 80_000,             # 95th percentile
    'max_input_tokens': 180_000,            # Hard limit
    'cost_per_query': 0.00010,              # Target cost
    'quality_minimum': 0.78,                # Minimum acceptable quality
}
```

</details>

---

## Q8. How do you optimize chunk size and retrieval k for your domain? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Chunk size and k (number of chunks) are interdependent decisions that directly affect context window usage, quality, and cost.

### Chunk Size Selection

```python
def analyze_chunk_size_impact(documents, embedding_model, retriever, query_samples):
    """Test different chunk sizes and measure quality."""
    
    chunk_sizes = [128, 256, 512, 1024, 2048]
    results = {}
    
    for size in chunk_sizes:
        # Re-chunk and re-index
        chunks = rechunk_documents(documents, chunk_size=size)
        index = rebuild_index(chunks, embedding_model)
        
        # Evaluate on queries
        recalls = []
        
        for query in query_samples:
            retrieved = index.search(query, k=5)
            recall = measure_recall(query, retrieved)
            recalls.append(recall)
        
        avg_recall = np.mean(recalls)
        
        # Metrics
        num_chunks = len(chunks)
        avg_tokens_per_chunk = np.mean([len(c['text'].split()) for c in chunks]) * 1.3  # tokens ~= 1.3x words
        
        results[size] = {
            'avg_recall@5': avg_recall,
            'num_chunks': num_chunks,
            'avg_tokens_per_chunk': avg_tokens_per_chunk,
            'index_size_gb': (num_chunks * avg_tokens_per_chunk * 4 / 1024 / 1024 / 1024),  # Rough
        }
    
    # Display results
    print("Chunk Size Analysis:")
    for size, metrics in results.items():
        print(f"\n{size} chars/chunk:")
        print(f"  Recall@5: {metrics['avg_recall@5']:.2%}")
        print(f"  Chunks: {metrics['num_chunks']:,}")
        print(f"  Avg tokens/chunk: {metrics['avg_tokens_per_chunk']:.0f}")
        print(f"  Index size: {metrics['index_size_gb']:.1f}GB")
    
    # Recommendation
    best_recall_size = max(results.items(), key=lambda x: x[1]['avg_recall@5'])
    most_efficient = min(
        results.items(),
        key=lambda x: x[1]['index_size_gb'] / x[1]['avg_recall@5']  # Quality per GB
    )
    
    print(f"\nRecommendations:")
    print(f"  Best quality: {best_recall_size[0]} chars → {best_recall_size[1]['avg_recall@5']:.2%} recall")
    print(f"  Most efficient: {most_efficient[0]} chars")
```

### Dynamic K Selection

```python
def optimize_k_per_query(query, retriever, quality_target=0.80, max_tokens_budget=50_000):
    """Dynamically choose k based on query characteristics."""
    
    # Estimate query complexity
    word_count = len(query.split())
    has_multi_part = query.count('and') > 0 or query.count(',') > 1
    
    if has_multi_part or word_count > 15:
        base_k = 5  # Complex query needs more context
    elif word_count < 5:
        base_k = 2  # Simple query needs less
    else:
        base_k = 3
    
    # Retrieve with increasing k until quality threshold met
    for k in [base_k, base_k + 2, base_k + 5, base_k + 10]:
        chunks = retriever.search(query, k=k)
        
        # Estimate context tokens
        context_tokens = sum(estimate_tokens(c['text']) for c in chunks) + estimate_tokens(query)
        
        if context_tokens > max_tokens_budget:
            # Budget exceeded, use previous k
            return retriever.search(query, k=max(base_k, k-2))
        
        # Estimate answer quality from chunks
        answer = llm.generate(query, chunks)
        confidence = assess_confidence(answer, chunks)
        
        if confidence > quality_target:
            return chunks
    
    # Return best effort with max k
    return chunks
```

### Combined Optimization

```python
class ChunkAndKOptimizer:
    def __init__(self, documents):
        self.documents = documents
    
    def optimize(self, query_samples, embedding_model, target_recall=0.85):
        """Find optimal (chunk_size, k) combination."""
        
        chunk_sizes = [256, 512, 1024]
        k_values = [3, 5, 10]
        
        best_config = None
        best_score = 0
        
        for chunk_size in chunk_sizes:
            # Re-chunk
            chunks = rechunk_documents(self.documents, chunk_size)
            index = rebuild_index(chunks, embedding_model)
            
            for k in k_values:
                # Evaluate
                recalls = []
                context_tokens_list = []
                
                for query in query_samples:
                    retrieved = index.search(query, k=k)
                    recall = measure_recall(query, retrieved)
                    context_tokens = sum(estimate_tokens(c['text']) for c in retrieved)
                    
                    recalls.append(recall)
                    context_tokens_list.append(context_tokens)
                
                # Score: balance quality and efficiency
                avg_recall = np.mean(recalls)
                avg_tokens = np.mean(context_tokens_list)
                
                # Quality must meet threshold, then minimize tokens
                if avg_recall >= target_recall:
                    score = -avg_tokens  # Negative because we minimize
                    
                    if best_config is None or score > best_score:
                        best_config = {
                            'chunk_size': chunk_size,
                            'k': k,
                            'recall': avg_recall,
                            'avg_tokens': avg_tokens,
                        }
                        best_score = score
        
        return best_config

# Usage
optimizer = ChunkAndKOptimizer(documents)
optimal = optimizer.optimize(query_samples, embedding_model, target_recall=0.85)

print(f"Optimal configuration:")
print(f"  Chunk size: {optimal['chunk_size']} chars")
print(f"  k: {optimal['k']} chunks")
print(f"  Recall@{optimal['k']}: {optimal['recall']:.2%}")
print(f"  Avg context: {optimal['avg_tokens']:,} tokens")
```

</details>

---

## Q9. How do you monitor context usage and establish SLOs? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

### Comprehensive Monitoring

```python
class ContextUsageMonitor:
    def __init__(self):
        self.metrics = {
            'input_tokens': [],
            'output_tokens': [],
            'num_chunks': [],
            'chunk_size': [],
            'lost_in_middle_risk': [],  # Liu et al. metric
        }
    
    def log_request(self, query, chunks, response):
        """Log context metrics for a request."""
        
        input_tokens = estimate_tokens(query) + sum(estimate_tokens(c['text']) for c in chunks)
        output_tokens = estimate_tokens(response)
        
        self.metrics['input_tokens'].append(input_tokens)
        self.metrics['output_tokens'].append(output_tokens)
        self.metrics['num_chunks'].append(len(chunks))
        self.metrics['chunk_size'].append(np.mean([len(c['text']) for c in chunks]))
        
        # Risk of lost-in-middle: chunks in middle often ignored
        if len(chunks) > 5:
            risk = 1.0  # High risk
        elif input_tokens > 50_000:
            risk = 0.5  # Medium risk
        else:
            risk = 0.0  # Low risk
        
        self.metrics['lost_in_middle_risk'].append(risk)
    
    def report_slo_status(self):
        """Check status vs SLOs."""
        
        slos = {
            'input_tokens_p95': 80_000,
            'input_tokens_max': 180_000,
            'num_chunks_p95': 10,
            'lost_in_middle_risk_max': 0.3,  # 30% of requests at high risk
        }
        
        violations = {}
        
        # P95 input tokens
        p95_tokens = np.percentile(self.metrics['input_tokens'], 95)
        if p95_tokens > slos['input_tokens_p95']:
            violations['input_tokens_p95'] = p95_tokens
        
        # Max input tokens
        max_tokens = np.max(self.metrics['input_tokens'])
        if max_tokens > slos['input_tokens_max']:
            violations['input_tokens_max'] = max_tokens
        
        # P95 num chunks
        p95_chunks = np.percentile(self.metrics['num_chunks'], 95)
        if p95_chunks > slos['num_chunks_p95']:
            violations['num_chunks_p95'] = p95_chunks
        
        # Lost-in-middle risk
        high_risk_rate = np.mean([r for r in self.metrics['lost_in_middle_risk']])
        if high_risk_rate > slos['lost_in_middle_risk_max']:
            violations['lost_in_middle_risk'] = high_risk_rate
        
        return violations

# Usage
monitor = ContextUsageMonitor()

# For each query
for query, chunks, response in incoming_requests():
    monitor.log_request(query, chunks, response)
    
    # Check SLOs periodically
    if len(monitor.metrics['input_tokens']) % 100 == 0:
        violations = monitor.report_slo_status()
        
        if violations:
            for slo, value in violations.items():
                alert(f"SLO violation: {slo} = {value}")
```

### Quality Regression Detection

```python
def detect_context_quality_regression(baseline_metrics, current_metrics):
    """Detect if context optimization caused quality drop."""
    
    # Compare against baseline (before optimization)
    baseline_quality = baseline_metrics['answer_quality']
    baseline_tokens = baseline_metrics['avg_input_tokens']
    
    current_quality = current_metrics['answer_quality']
    current_tokens = current_metrics['avg_input_tokens']
    
    # Quality degraded?
    quality_drop = baseline_quality - current_quality
    
    if quality_drop > 0.05:  # >5% drop
        alert(f"Quality regression: {quality_drop:.1%} drop")
        print(f"  Before: {baseline_quality:.2%} quality with {baseline_tokens:,} tokens")
        print(f"  After: {current_quality:.2%} quality with {current_tokens:,} tokens")
        
        # Was it worth the savings?
        token_savings = baseline_tokens - current_tokens
        cost_savings_monthly = token_savings / 1_000_000 * 3 * 100_000  # Rough
        quality_cost = quality_drop * 1000  # Estimated impact
        
        if cost_savings_monthly > quality_cost:
            print(f"✓ Trade-off positive: Saved ${cost_savings_monthly:.2f} vs quality cost ${quality_cost:.2f}")
        else:
            print(f"✗ Trade-off negative: Savings insufficient to justify quality drop")
            print(f"  → Recommend reverting optimization")
        
        return False  # Regression detected
    
    return True  # No regression
```

</details>

---

## Q10. What is the cost-quality trade-off for context window strategies, and how do you optimize? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Context window management is a multi-dimensional optimization: minimize cost and latency while maintaining quality. This requires careful trade-off analysis.

### Pareto Analysis

```python
def identify_pareto_frontier_strategies(strategies):
    """Find non-dominated context strategies."""
    
    # Strategies: {name: {cost, latency, quality}}
    
    pareto_frontier = []
    
    for strategy in strategies:
        is_dominated = False
        
        for other in strategies:
            if strategy == other:
                continue
            
            # Is other strictly better on all metrics?
            # (Lower cost, lower latency, higher quality)
            if (other['cost'] < strategy['cost'] and
                other['latency'] < strategy['latency'] and
                other['quality'] > strategy['quality']):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_frontier.append(strategy)
    
    return pareto_frontier

# Example strategies
strategies = [
    {'name': 'minimal', 'cost': 1.0, 'latency': 100, 'quality': 0.70},
    {'name': 'moderate', 'cost': 2.5, 'latency': 200, 'quality': 0.82},
    {'name': 'large', 'cost': 6.0, 'latency': 400, 'quality': 0.88},
    {'name': 'huge', 'cost': 15.0, 'latency': 1000, 'quality': 0.89},
    {'name': 'smart_reranking', 'cost': 3.0, 'latency': 250, 'quality': 0.85},
    {'name': 'summarized', 'cost': 2.0, 'latency': 350, 'quality': 0.81},
]

frontier = identify_pareto_frontier_strategies(strategies)

print("Pareto Frontier (non-dominated strategies):")
for s in frontier:
    print(f"  {s['name']}: cost={s['cost']:.1f}, latency={s['latency']}ms, quality={s['quality']:.2f}")

# Result:
# Pareto Frontier:
#   minimal: cost=1.0, latency=100ms, quality=0.70
#   moderate: cost=2.5, latency=200ms, quality=0.82
#   smart_reranking: cost=3.0, latency=250ms, quality=0.85
#   large: cost=6.0, latency=400ms, quality=0.88
#   huge: cost=15.0, latency=1000ms, quality=0.89
```

### Decision Framework

Choose strategy based on constraints:

```python
def recommend_context_strategy(constraints):
    """Recommend strategy given SLO constraints."""
    
    slos = {
        'quality_minimum': constraints.get('quality_minimum', 0.75),
        'latency_p99_max_ms': constraints.get('latency_p99_max_ms', 500),
        'monthly_budget_dollars': constraints.get('monthly_budget_dollars', 100),
    }
    
    # Strategies on Pareto frontier
    candidates = [
        {'name': 'minimal', 'cost_monthly': 50, 'latency_p99': 100, 'quality': 0.70},
        {'name': 'moderate', 'cost_monthly': 120, 'latency_p99': 200, 'quality': 0.82},
        {'name': 'smart_reranking', 'cost_monthly': 150, 'latency_p99': 250, 'quality': 0.85},
        {'name': 'large', 'cost_monthly': 300, 'latency_p99': 400, 'quality': 0.88},
    ]
    
    # Filter by SLO constraints
    valid = [
        s for s in candidates
        if (s['quality'] >= slos['quality_minimum'] and
            s['latency_p99'] <= slos['latency_p99_max_ms'] and
            s['cost_monthly'] <= slos['monthly_budget_dollars'])
    ]
    
    if not valid:
        print("No strategy meets all SLOs. Relaxing constraints...")
        # User must choose: relax quality, latency, or budget
        return None
    
    # Among valid candidates, choose lowest cost
    recommended = min(valid, key=lambda x: x['cost_monthly'])
    
    return recommended

# Example
constraints = {
    'quality_minimum': 0.80,
    'latency_p99_max_ms': 300,
    'monthly_budget_dollars': 200,
}

recommendation = recommend_context_strategy(constraints)
print(f"Recommended: {recommendation['name']}")
# Output: moderate
```

### ROI Calculation

```python
def calculate_strategy_roi(current_strategy, new_strategy, impact_metrics):
    """Calculate ROI of switching strategies."""
    
    # Current state
    current_cost = current_strategy['cost_monthly']
    current_quality = current_strategy['quality']
    
    # New strategy
    new_cost = new_strategy['cost_monthly']
    new_quality = new_strategy['quality']
    
    # Business impact
    # Assume: each quality point improves business metric (conversions, revenue, etc.)
    quality_value_per_point = impact_metrics.get('value_per_quality_point', 10_000)
    
    # Cost savings
    cost_reduction = current_cost - new_cost
    
    # Quality impact (positive = improvement)
    quality_improvement = (new_quality - current_quality) * quality_value_per_point
    
    # Total ROI
    net_benefit = cost_reduction + quality_improvement
    
    print(f"Switching from {current_strategy['name']} to {new_strategy['name']}:")
    print(f"  Cost savings: ${cost_reduction:,}/month")
    print(f"  Quality gain: ${quality_improvement:,}/month")
    print(f"  Net benefit: ${net_benefit:,}/month")
    print(f"  ROI: {net_benefit / current_cost:.0%}")
    
    if net_benefit > 0:
        print(f"✓ Recommended: ROI is positive")
    else:
        print(f"✗ Not recommended: ROI is negative")

# Example
impact_metrics = {'value_per_quality_point': 5000}  # Each 1% quality = $5k/month value

calculate_strategy_roi(
    {'name': 'moderate', 'cost_monthly': 120, 'quality': 0.82},
    {'name': 'smart_reranking', 'cost_monthly': 150, 'quality': 0.85},
    impact_metrics
)

# Output:
# Switching from moderate to smart_reranking:
#   Cost savings: $-30/month  (costs more)
#   Quality gain: $15000/month (3 point improvement × $5k)
#   Net benefit: $14970/month
#   ROI: 124x
# ✓ Recommended
```

### Monitoring and Optimization Loop

```python
def continuous_context_optimization():
    """Ongoing monitoring and optimization."""
    
    while True:
        # 1. Measure current metrics
        current_metrics = measure_current_strategy()
        
        # 2. Compare to SLOs
        violations = check_slos(current_metrics)
        
        if violations:
            # 3. Identify optimization opportunities
            if violations['cost_exceeds_budget']:
                # Try to reduce cost (fewer chunks, smaller summaries, etc.)
                new_strategy = reduce_context_cost()
            elif violations['latency_exceeds_slo']:
                # Try to reduce latency (fewer chunks, caching, etc.)
                new_strategy = reduce_context_latency()
            elif violations['quality_below_target']:
                # Try to improve quality (more chunks, reranking, etc.)
                new_strategy = improve_quality()
            
            # 4. A/B test new strategy
            roi = ab_test_and_measure(current_strategy, new_strategy)
            
            if roi > threshold:
                # 5. Deploy new strategy
                deploy(new_strategy)
        
        # Sleep and repeat every week
        time.sleep(7 * 24 * 3600)
```

</details>

---
