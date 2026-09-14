# 32 — Few-Shot Example RAG (PEARL / Example-Augmented Prompting)

> Retrieves demonstration examples (query→answer pairs) instead of documents — teaches the LLM the expected output format and reasoning pattern via in-context examples.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
New Query
    │
    ▼
Example Retriever (search vector store of (query, answer) demonstration pairs)
    │
    ▼
Top-k Similar Demonstrations selected
    │
    ▼
Prompt Assembler (builds few-shot prompt: demonstrations + new query)
    │
    ▼
Generator (LLM produces the answer, conditioned on the retrieved examples)
```

### Key Components

| Component | Responsibility |
|---|---|
| Demonstration Store | Vector DB of (query, answer) pairs, indexed on the query side |
| Example Retriever | Finds the top-k demonstrations most semantically similar to the new query |
| Prompt Assembler | Formats retrieved demonstrations and the new query into a single few-shot prompt |
| Generator | LLM that produces the answer, following the pattern shown by the retrieved examples rather than raw documents |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Embedding model | sentence-transformers (e.g. `all-MiniLM-L6-v2`) |
| Vector store | FAISS, Chroma |
| Example selection | MMR (Maximal Marginal Relevance) for relevance + diversity |
| Automated optimization | DSPy for automated example selection and prompt optimization |

---

## Q1. What is Few-Shot Example RAG and how does it differ from standard document RAG? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Standard RAG** retrieves *documents* — passages that contain factual information the LLM uses to answer the query.

**Few-Shot Example RAG** retrieves *examples* — prior (query, answer) pairs that show the LLM the expected output format, style, and reasoning pattern for the current query. The LLM then generates a new answer that follows the demonstrated pattern.

```
Standard RAG:
  User: "What is the capital of France?"
  Retrieved: [Document about France containing "Paris is the capital..."]
  LLM uses document as reference → "The capital of France is Paris."

Few-Shot Example RAG:
  User: "Translate 'hello' to Spanish."
  Retrieved examples:
    - ("Translate 'dog' to Spanish", "perro")
    - ("Translate 'house' to Spanish", "casa")
    - ("Translate 'water' to Spanish", "agua")
  LLM sees pattern: translate single word → single word translation → "hola"
```

**When few-shot example retrieval outperforms document retrieval:**

| Scenario | Standard RAG | Few-Shot Example RAG |
|----------|--------------|---------------------|
| Factual Q&A | ✓ Excellent | ✗ Not applicable |
| Code generation (follow project style) | Limited | ✓ Excellent |
| Structured output (follow schema) | Possible | ✓ Excellent |
| Few-shot classification | Poor | ✓ Excellent |
| Chain-of-thought tasks | Limited | ✓ Excellent |
| Text-to-SQL (follow naming conventions) | Limited | ✓ Excellent |

**The key insight:** For tasks where the LLM needs to see *what good output looks like* rather than *what the facts are*, retrieving examples is more informative than retrieving documents.

</details>

---

## Q2. How is the example datastore built and queried in PEARL? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**PEARL** (Prompting via Example-based and Adaptive Retrieval from a Library) introduced a systematic approach to building and querying an example library.

**Building the example datastore:**

```python
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, json

model = SentenceTransformer("all-MiniLM-L6-v2")

# Example library: list of (query, answer) pairs
examples = [
    {"query": "Write a Python function to reverse a string",
     "answer": "def reverse_string(s: str) -> str:\n    return s[::-1]"},
    {"query": "Write a Python function to check if a number is prime",
     "answer": "def is_prime(n: int) -> bool:\n    if n < 2: return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0: return False\n    return True"},
    # ... thousands more examples
]

# Embed the QUERY side of each example (not the answer)
queries = [ex["query"] for ex in examples]
embeddings = model.encode(queries, normalize_embeddings=True)

# Build FAISS index over query embeddings
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings.astype(np.float32))
```

**Why embed the query side?**

You retrieve examples whose *questions* are semantically similar to the current question. An example whose question matches the current question will demonstrate a similar reasoning pattern.

**Querying at inference time:**

```python
def retrieve_examples(current_query: str, k: int = 3) -> list[dict]:
    q_emb = model.encode([current_query], normalize_embeddings=True)
    scores, indices = index.search(q_emb.astype(np.float32), k)
    return [examples[i] for i in indices[0]]

def few_shot_rag(query: str) -> str:
    retrieved = retrieve_examples(query, k=3)
    
    # Format as few-shot prompt
    few_shot_block = ""
    for ex in retrieved:
        few_shot_block += f"Q: {ex['query']}\nA: {ex['answer']}\n\n"
    
    prompt = f"{few_shot_block}Q: {query}\nA:"
    
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

**Output with retrieved examples:**

```
Q: Write a Python function to reverse a string
A: def reverse_string(s: str) -> str:
       return s[::-1]

Q: Write a Python function to check if a number is prime
A: def is_prime(n: int) -> bool:
       ...

Q: Write a Python function to count vowels in a string
A: [LLM follows the pattern: function signature, docstring style, implementation]
```

</details>

---

## Q3. How do you select which examples to retrieve to maximize in-context learning? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Naive nearest-neighbor retrieval can return near-identical examples. Good example selection requires balancing **relevance** and **diversity**.

**Anti-pattern: retrieving near-duplicates**

```
Query: "Sort a list of integers in Python"
Retrieved:
  - "Sort a list of numbers in Python"   ← near-duplicate
  - "Sort a Python list"                  ← near-duplicate  
  - "Sort integers using Python"          ← near-duplicate

All 3 examples look the same → no additional information for the LLM
```

**Better: Diverse but relevant examples**

```
Query: "Sort a list of integers in Python"
Retrieved:
  - "Sort a list of strings by length"     ← sorts, different key function
  - "Sort a list of dicts by a field"       ← sorts, complex key
  - "Sort a list of integers in reverse"   ← sorts integers, different parameter
```

**MMR-based example selection:**

```python
def select_examples_mmr(
    query: str,
    candidate_examples: list[dict],
    k: int = 3,
    lambda_: float = 0.7
) -> list[dict]:
    """Maximal Marginal Relevance for example selection."""
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    cand_embs = model.encode([ex["query"] for ex in candidate_examples],
                              normalize_embeddings=True)
    
    selected = []
    selected_embs = []
    
    while len(selected) < k:
        scores = []
        for i, (ex, emb) in enumerate(zip(candidate_examples, cand_embs)):
            if i in [candidate_examples.index(s) for s in selected]:
                continue
            relevance = float(q_emb @ emb)
            if not selected_embs:
                redundancy = 0.0
            else:
                redundancy = max(float(emb @ s_emb) for s_emb in selected_embs)
            mmr_score = lambda_ * relevance - (1 - lambda_) * redundancy
            scores.append((i, mmr_score))
        
        best_idx = max(scores, key=lambda x: x[1])[0]
        selected.append(candidate_examples[best_idx])
        selected_embs.append(cand_embs[best_idx])
    
    return selected
```

**Coverage-based selection for chain-of-thought:**

For tasks requiring multi-step reasoning, select examples that cover different *reasoning patterns*:

```python
# Cluster examples by reasoning type first, then retrieve one from each cluster
# E.g., for math word problems:
# Cluster 1: single arithmetic step
# Cluster 2: multi-step with unit conversion
# Cluster 3: word problem requiring extraction of key values
```

**Ordering matters:**

Research (Min et al., 2022) shows that placing the most similar example **last** (closest to the actual query) yields the best performance — the LLM's attention pattern means recent context is weighed more heavily.

</details>

---

## Q4. How is Few-Shot Example RAG used for text-to-SQL generation? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Text-to-SQL is one of the highest-value applications: retrieve SQL examples that match the schema and query pattern, so the LLM learns the exact table names, column names, and join patterns in use.

```python
SQL_EXAMPLES = [
    {
        "question": "How many orders were placed last month?",
        "sql": "SELECT COUNT(*) FROM orders WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') AND created_at < DATE_TRUNC('month', CURRENT_DATE)"
    },
    {
        "question": "What is the total revenue by product category?",
        "sql": "SELECT p.category, SUM(oi.quantity * oi.unit_price) AS revenue FROM order_items oi JOIN products p ON oi.product_id = p.id GROUP BY p.category ORDER BY revenue DESC"
    },
    # ...
]

def text_to_sql(question: str, schema: str) -> str:
    examples = retrieve_examples(question, k=3)
    
    few_shot = "\n".join(
        f"-- Question: {ex['question']}\n{ex['sql']}\n"
        for ex in examples
    )
    
    prompt = f"""You are a SQL expert. Given a database schema and example queries,
write a SQL query for the new question.

Schema:
{schema}

Example queries:
{few_shot}

-- Question: {question}
-- SQL:"""
    
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

**Why this outperforms a static few-shot prompt:**
- Static prompts use the same 3–5 examples for every query → examples may be irrelevant
- Dynamic retrieval finds examples with the same JOIN patterns, aggregations, or filters as the current question
- When the schema has hundreds of tables, retrieved examples implicitly teach which tables are relevant

</details>

---

## Q5. How do you combine document RAG and example RAG in the same pipeline? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The two types of retrieval are complementary and can be combined in the same context window.

```
User Query
    │
    ├──► Document Retrieval    → "What are the facts?" (retrieved passages)
    │
    └──► Example Retrieval     → "What should the output look like?" (demonstrations)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ Prompt structure:                                    │
│                                                     │
│ [Few-shot examples — teach format/style]            │
│ Example 1: Q + A                                    │
│ Example 2: Q + A                                    │
│                                                     │
│ [Retrieved documents — provide facts]               │
│ Document 1: relevant passage                        │
│ Document 2: relevant passage                        │
│                                                     │
│ [Current query]                                     │
└─────────────────────────────────────────────────────┘
```

**Implementation:**

```python
def hybrid_rag(query: str) -> str:
    # Retrieve both types
    doc_results = doc_retriever.retrieve(query, k=3)
    example_results = retrieve_examples(query, k=2)
    
    # Build combined prompt
    examples_block = ""
    for ex in example_results:
        examples_block += f"Q: {ex['query']}\nA: {ex['answer']}\n\n"
    
    docs_block = "\n\n".join(
        f"[Document {i+1}]: {doc.page_content}"
        for i, doc in enumerate(doc_results)
    )
    
    prompt = f"""Examples of the expected answer format:
{examples_block}
---
Reference documents (use these for factual content):
{docs_block}
---
Question: {query}
Answer (following the format shown in the examples):"""
    
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

**Use cases for combined retrieval:**

- **Technical documentation with style guide**: Documents provide API facts; examples show the expected explanation format
- **Medical report generation**: Documents provide clinical guidelines; examples show the expected SOAP note format
- **Code review**: Documents provide security best practices; examples show what a good code review comment looks like

</details>

---

## Q6. Walk through the Few-Shot Example RAG architecture end-to-end. `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

```
New Query
    │
    ▼
Example Retriever (search vector store of (query, answer) demonstration pairs)
    │
    ▼
Top-k Similar Demonstrations selected
    │
    ▼
Prompt Assembler (builds few-shot prompt: demonstrations + new query)
    │
    ▼
Generator (LLM produces the answer, conditioned on the retrieved examples)
```

The query side of each stored demonstration is embedded and indexed exactly like a document would be in standard RAG — the architectural machinery (embed, index, ANN search) is identical. What differs is entirely what's stored and how the retrieved result is used: instead of passages that get cited as factual support, retrieved items are complete (question, answer) pairs that get formatted as in-context examples, teaching the generator a pattern to follow rather than a fact to report. This is why Few-Shot Example RAG can reuse essentially all of standard RAG's retrieval infrastructure while solving a fundamentally different problem (Q7).

</details>

---

## Q7. What is the single distinctive mechanism that separates Few-Shot Example RAG from standard document RAG? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The distinctive mechanism is retrieving **demonstrations of the task being performed** rather than **evidence about the world**. Standard RAG answers "what are the facts relevant to this query" — the retrieved content is consumed as reference material to cite or summarize. Few-Shot Example RAG answers a different question entirely: "what does correct output for a query like this one actually look like" — the retrieved content is consumed as a pattern to imitate (format, style, reasoning structure), not as information to report.

This reframing is why the same underlying retrieval mechanism (embed, index, similarity search) produces such different value depending on what's indexed: for tasks with a clear right factual answer (capital of France), examples add nothing since there's no "format" to learn; for tasks where correctness is largely about *how* the output is structured (SQL following schema conventions, code following project style, structured extraction following a schema), examples are far more informative than any document could be, because no document states "here is what a well-formed answer to this type of question looks like" as directly as a worked example does.

</details>

---

## Q8. How does Few-Shot Example RAG compare to RAG-Fusion (#18) and HyDE (#22)? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

RAG-Fusion (#18) and HyDE (#22) both modify the *query side* of standard document retrieval to improve recall against a document corpus — RAG-Fusion generates multiple query reformulations and merges their results; HyDE embeds a hypothetical answer instead of the raw query to close the query-document vocabulary gap. Both still retrieve documents as factual evidence; they change *how* the query is formed, not *what kind of thing* is retrieved.

Few-Shot Example RAG changes what's indexed and retrieved altogether — demonstrations instead of documents — which makes it orthogonal to, not competing with, RAG-Fusion or HyDE. A production system generating SQL, for instance, could use Few-Shot Example RAG for its demonstration layer while separately using HyDE-style query reformulation if it also needs to retrieve schema documentation as supporting context (Q18's combined-retrieval pattern) — the two techniques address different retrieval problems and compose rather than substitute for each other.

</details>

---

## Q9. What is the research origin of PEARL and example-based retrieval for in-context learning? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

PEARL (Prompting via Example-based and Adaptive Retrieval from a Library) formalized the pattern of building a searchable library of (query, answer) demonstrations and dynamically retrieving the most relevant ones per query, rather than using a fixed, hand-picked set of few-shot examples in every prompt. This builds on a broader line of in-context-learning research establishing that *which* examples appear in a few-shot prompt, and in what order, materially affects output quality — notably Min et al. (2022), whose finding that placing the most similar example last (closest to the query) improves performance is the ordering principle referenced in Q3 and expanded in Q14.

The broader shift this research reflects: early few-shot prompting used the same static example set for every query, discovered through manual trial and error. PEARL-style retrieval automates and personalizes that selection per query, which is the same "static curation to dynamic retrieval" shift that defines RAG as a category more broadly, applied specifically to the choice of in-context demonstrations rather than factual context.

</details>

---

## Q10. What are the key tuning knobs for Few-Shot Example RAG, and how do you choose them? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Knob | Effect | Starting point |
|---|---|---|
| `k` (number of retrieved examples) | More examples give the model more pattern signal but consume more context and risk diluting attention across examples | 3-5 for most tasks; fewer for very long demonstrations (full code files), more for short ones (single-word translations) |
| MMR `lambda_` (relevance vs. diversity, Q3) | Higher favors pure relevance (risking near-duplicates); lower favors diversity (risking irrelevant examples) | 0.7 is a reasonable default, tuned against the near-duplicate anti-pattern in Q3 |
| Embedding model | Determines what "similar query" means — a general-purpose model may miss domain-specific structural similarity (e.g., two SQL queries with the same JOIN pattern but different table names) | Start with a general-purpose model (`all-MiniLM-L6-v2`); fine-tune on your domain if structural similarity matters more than surface wording similarity |
| Example ordering (Q3, Q14) | Position within the prompt affects how much attention each example receives | Place the most similar/highest-confidence example last, closest to the query |

The embedding model choice is the least obvious but often most consequential knob: a general-purpose semantic embedding model is tuned to recognize topical similarity in natural language, which doesn't always align with "structural similarity" for tasks like code or SQL generation, where two queries can be topically unrelated but structurally identical (same control-flow pattern, same JOIN structure) in exactly the way that matters for demonstration usefulness.

</details>

---

## Q11. How do you evaluate whether retrieved examples are actually improving output quality over a static few-shot baseline? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Build a task-specific golden set (queries with known-correct outputs, e.g. working SQL queries or correctly-formatted extractions) and compare three conditions: (1) zero-shot (no examples at all); (2) static few-shot (the same fixed 3-5 examples for every query, chosen manually); (3) dynamic retrieval (Q2's PEARL-style per-query retrieval). Measure task-specific correctness (does the generated SQL execute and return the right result; does the generated code pass tests; does the extraction match the schema) rather than a generic text-similarity metric, since the whole point of examples is influencing *structural* correctness that a surface-level text comparison won't capture well.

Segment by query type if the task has meaningfully different sub-patterns (simple vs. complex SQL joins, short vs. long code snippets) — dynamic retrieval's advantage over static few-shot should be largest exactly where the task's correct pattern varies most across queries, and smallest (or even negative, given the added retrieval latency) where a single static example set already covers the pattern space adequately. This segmentation is what feeds the Q15 decision gate on whether the retrieval infrastructure is worth the complexity for a given task.

</details>

---

## Q12. What is the characteristic failure mode when retrieved examples are near-duplicates or off-pattern, and how do you detect it in production? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Q3's near-duplicate anti-pattern (retrieving three examples that are all trivial rephrasings of the same underlying pattern) has a subtle production signature: the system doesn't error or obviously fail — it just provides less pattern diversity than `k` examples suggests, silently degrading the demonstration quality for any query whose correct handling actually requires seeing a *different* pattern than the one over-represented in the retrieved set. A related but distinct failure is off-pattern retrieval: examples that are semantically similar in topic to the query but demonstrate an irrelevant structural pattern (e.g., retrieving a single-table SQL example for a query that actually needs a multi-table join), which can actively mislead the generator toward the wrong output shape rather than just failing to help.

**Detection:** log the pairwise similarity among each query's retrieved example set — a consistently high average pairwise similarity across production queries is the near-duplicate signature and suggests MMR's diversity term (`lambda_`, Q10) needs adjusting. For off-pattern retrieval, track task-specific output correctness (Q11) segmented by whether the retrieved examples' *structural* features (not just semantic similarity score) actually matched the query's requirements — this requires a task-specific structural-similarity check (e.g., comparing SQL AST shape, not just embedding cosine similarity) as a secondary signal alongside the primary retrieval score.

</details>

---

## Q13. How do you build and maintain an example library over time? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

An initial example library is typically seeded from a curated set of known-good (query, answer) pairs — hand-written for a new task, or mined from historical logs where output quality was verified (a code snippet that passed review, an SQL query that was confirmed correct). Maintaining it over time requires deliberate curation, not just accumulation:

1. **Incremental additions from production** — a query that received a highly-rated or manually-corrected answer is a candidate for library inclusion, but should be reviewed before addition rather than auto-added, since a plausible-looking but subtly wrong example poisons every future query that retrieves it (Q19).
2. **Deduplication** — as the library grows, near-duplicate examples (Q12) accumulate naturally; periodically clustering the library and pruning redundant examples keeps both retrieval quality and index size in check.
3. **Staleness management** — an example demonstrating an outdated convention (an old API signature, a deprecated coding pattern) needs to be retired or updated when the underlying convention changes, exactly as a document RAG corpus needs freshness management (#35) — the difference is that a stale example doesn't just provide outdated information, it can actively teach the generator to produce outdated-pattern output.
4. **Coverage auditing** — periodically check whether the library has thin coverage for any query category that production traffic actually needs (few or no examples retrievable above a relevance threshold), which signals a gap to fill with new curated examples rather than letting the retriever silently fall back to poor-fit results.

</details>

---

## Q14. How does example ordering and prompt placement affect in-context learning performance? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Research (Min et al., 2022) found that placing the most similar retrieved example **last** — immediately before the new query — yields the best performance, attributed to LLMs weighing recent context more heavily than earlier context in a prompt (a milder version of the same attention-distribution phenomenon behind "lost in the middle," #43 Q14). This has a direct, actionable implication for prompt assembly: don't simply concatenate retrieved examples in retrieval-score order or in arbitrary order — explicitly sort them so the highest-relevance example sits closest to the query, even if it wasn't the highest-scoring example by MMR's combined relevance-diversity score (Q10).

This also interacts with `k`: as more examples are added, the ones placed earlier in the prompt contribute progressively less influence on the final output, which means beyond some point, adding more examples produces diminishing returns not just from redundancy (Q12) but from positional attention decay — a smaller number of well-ordered, high-quality examples can outperform a larger number where most examples sit in a positionally weak part of the prompt. This is a distinct consideration from Q10's relevance/diversity trade-off, and both should be tuned together: MMR selects *which* examples, ordering determines *how much influence* each selected example actually has once assembled into the final prompt.

</details>

---

## Q15. How would you build a decision-gate benchmark to decide whether dynamic example retrieval is worth it over a static few-shot prompt? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Dynamic retrieval adds real infrastructure (an example library, an embedding pipeline, a vector index, ongoing curation per Q13) relative to hard-coding 3-5 examples directly into a prompt template — the decision should be measured against that added complexity, not assumed:

```
1. Build a golden set (Q11) covering the realistic diversity of query
   patterns the task will face in production.

2. Baseline: hand-pick the best static few-shot set you reasonably can
   (this itself takes real effort -- don't compare against a lazy
   static baseline, or the comparison is unfair) and measure task
   correctness on the full golden set.

3. Candidate: measure the same metric with dynamic retrieval (Q2) at
   a reasonable k and MMR lambda.

4. Segment the comparison by query-pattern diversity: if the golden
   set's query patterns cluster into just 2-3 distinct shapes, a static
   set covering those shapes may perform comparably to dynamic
   retrieval; if patterns are highly varied (hundreds of distinct SQL
   join/filter combinations), dynamic retrieval's advantage should be
   large and clear.

5. Gate: adopt dynamic retrieval only if it clears a meaningful accuracy
   improvement over the best achievable static baseline -- for narrow,
   low-diversity tasks (a fixed classification schema with 5 categories),
   a well-chosen static few-shot set is often good enough that the
   retrieval infrastructure's maintenance cost (Q13) isn't justified.
```

The key discipline is comparing against a *genuinely good* static baseline, not a strawman — the temptation to compare dynamic retrieval against a lazily-chosen static set overstates the case for retrieval infrastructure that a well-curated static prompt might match at a fraction of the operational complexity.

</details>

---

## Q16. What is the cost and latency overhead of Few-Shot Example RAG at scale, and how do you control it? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The retrieval step itself (embed query, ANN search, format prompt) is essentially the same cost profile as standard document RAG's retrieval step — cheap relative to the generation call. The cost dimension specific to this architecture is **prompt length**: k retrieved examples, each a full (query, answer) pair, typically consume more tokens than k retrieved document chunks would (a full code example or SQL query is often longer than an equivalently-sized document passage), directly increasing the per-query generation cost proportional to `k x average_example_length`.

Illustrative comparison for a code-generation task: 3 retrieved examples averaging 150 tokens each add 450 tokens to every prompt versus a static 3-example baseline (identical cost) — the cost delta versus static few-shot is actually zero at the generation step; the *added* cost relative to static few-shot is entirely in retrieval infrastructure (embedding the library, hosting the vector index, the embedding call per query) rather than generation. This is a different cost profile than document RAG vs. no-RAG, where retrieval materially reduces required generation-time knowledge; here, retrieval's cost is almost entirely about *selecting better examples*, not about reducing token consumption — the token cost is comparable to any few-shot prompting approach with the same `k`.

**Controls:** cap `k` based on the marginal-value analysis from Q14 (diminishing returns past a positionally-effective count); use a smaller/faster embedding model for the retrieval step specifically, since it's a high-frequency, low-complexity operation relative to the generation call; and cache retrieval results for repeated or near-identical queries, exactly as any RAG system would.

</details>

---

## Q17. What security and trust risks does a crowd-sourced or production-fed example library introduce? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

- **Poisoned demonstrations** — since retrieved examples directly shape the generator's *output pattern* (not just its factual grounding), a maliciously or erroneously incorrect example in the library (a code snippet with a subtle security vulnerability, an SQL query with an injection-prone pattern) doesn't just risk being cited incorrectly — it actively teaches the generator to reproduce the flawed pattern in every future output that retrieves it, a more direct and consequential failure than a poisoned document in standard RAG, where a bad document is one piece of evidence the generator weighs rather than a pattern it imitates.
- **Auto-ingestion from unreviewed production traffic** (Q13's incremental-addition risk) — a pipeline that automatically adds highly-rated production outputs to the library without human review can incorporate an example that looked good to whatever rating signal triggered inclusion (a thumbs-up, a low-perplexity score) but is actually subtly wrong in a way the rating signal didn't catch, compounding over time as more flawed examples get added and retrieved for similar future queries.
- **Convention drift exploited adversarially** — if an attacker can influence what gets added to the library (a crowd-sourced or user-submitted example system), they could seed examples that appear helpful but bias output toward an insecure pattern (e.g., an example demonstrating SQL query construction via string concatenation rather than parameterized queries), which then propagates into every future generation that retrieves it.
- **Example-library membership inference** — since retrieved examples appear verbatim in prompts (and potentially in logs), a library built from real user queries and answers carries the same data-exposure risk as any RAG corpus containing user-derived content, requiring the same PII/sensitivity screening applied to any indexed corpus.

Mitigation: require human review before any example enters the library from an automated pipeline (never fully auto-ingest), periodically audit a sample of the library for correctness and convention adherence (treating library quality as an ongoing maintenance responsibility, not a one-time curation task), and apply the same source-trust principles used for document corpora if the library accepts contributions from less-trusted sources.

</details>

---

## Q18. Design a Few-Shot Example RAG system combined with document retrieval for a customer support assistant. `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Requirements:** responses need both a consistent, on-brand tone/format and factually accurate, current information (product details, policy specifics) — neither examples alone nor documents alone fully satisfy both needs (Q1's table).

```
1. Example library: curated from past highly-rated support tickets,
   reviewed before inclusion (Q17) -- each example is a (customer
   query, well-formatted resolution) pair demonstrating the target
   tone, structure, and level of detail.

2. Document corpus: standard product documentation, policy pages,
   troubleshooting guides -- the factual source of truth, kept fresh
   via normal document RAG update discipline.

3. Combined retrieval (Q5's hybrid pattern): retrieve top-2 examples
   (format/tone guidance) and top-3 documents (factual grounding) per
   query, assembled into a single prompt with examples establishing
   the expected response shape and documents providing the specific
   facts to fill into that shape.

4. Library-document consistency check: periodically verify that
   examples still reflect CURRENT policy -- an example demonstrating
   a resolution for an outdated policy is actively harmful here (Q13's
   staleness risk), more so than a stale example in a lower-stakes
   domain, since customer-facing incorrect policy statements carry
   real business/compliance risk.

5. Evaluation (Q11): segment by whether responses are tone-correct
   (examples doing their job) AND factually correct (documents doing
   theirs) as two separate scored dimensions -- a response can fail
   on either independently, and conflating them into one quality score
   would hide which component of the combined system needs attention.
```

The key design insight is treating tone/format correctness and factual correctness as genuinely separate quality dimensions requiring separate retrieval sources and separate evaluation — this is precisely the insight Q1's comparison table and Q7's "facts vs. patterns" distinction establish, applied concretely to a use case where both dimensions matter simultaneously.

</details>

---

## Q19. What happens when two retrieved examples demonstrate conflicting formats or conventions, and how do you handle it? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

If the example library contains inconsistent conventions (some SQL examples using `snake_case` aliases, others `camelCase`; some code examples with docstrings, others without) — often the natural result of a library accumulated over time from multiple contributors or eras of a codebase (Q13's staleness/drift risk) — retrieving a mixed set for one query presents the generator with contradictory patterns to imitate simultaneously, with no principled way to know which convention is currently preferred. The result is inconsistent output: the generator might blend both conventions incoherently, or arbitrarily follow whichever example happened to be positioned most influentially (Q14's ordering effect), producing unpredictable formatting from otherwise-similar queries.

**Detection:** this shows up as inconsistency in generated output style across similar queries — segment production outputs by which examples were retrieved and check for format/convention divergence correlated with which "era" or "cluster" of the library was retrieved from. **Mitigation:** (1) tag examples with a convention/version identifier at curation time, and either exclude or down-weight retrieval of deprecated-convention examples once a newer convention is established, rather than leaving the library's full history equally retrievable; (2) periodically re-normalize older examples to current conventions rather than only adding new ones on top of an inconsistent base; (3) for a codebase or convention undergoing active migration, consider explicitly retrieving only from the target (new) convention's example subset during the transition period, accepting reduced example diversity in exchange for consistent output.

</details>

---

## Q20. What are the limitations of Few-Shot Example RAG, and how might the field evolve? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Current limitations: (1) **the technique only helps for pattern-following tasks, not factual tasks** (Q1, Q7) — it's not a general-purpose retrieval upgrade, and misapplying it to a task that actually needs facts (not patterns) provides no benefit; (2) **library quality is a continuous maintenance burden** (Q13, Q19) — unlike a document corpus where staleness mainly risks outdated facts, a stale or inconsistent example library actively teaches wrong patterns, raising the maintenance stakes; (3) **retrieval selects for surface/embedding similarity, not necessarily structural similarity** (Q10) — general-purpose embedding models don't always recognize the specific structural properties (code control flow, SQL join shape) that actually determine whether an example is a useful demonstration; (4) **poisoning risk is more direct than in document RAG** (Q17) since examples are imitated, not just cited.

Likely evolution: task-specific, structurally-aware retrieval (embedding models or similarity functions trained specifically to recognize code/SQL/schema structural similarity rather than general semantic similarity) replacing today's general-purpose embedding defaults; tighter integration with automated prompt-optimization frameworks (DSPy, already referenced in this file's tools table) that jointly optimize example selection and prompt structure rather than treating retrieval and prompt assembly as separate, independently-tuned stages; and more systematic library lifecycle tooling (automated staleness detection, convention-consistency auditing, Q13/Q19) as the pattern matures from a research technique into standard production infrastructure the way document RAG's own tooling has.

</details>

---

## Real-World Applications

- **GitHub Copilot**: Retrieves similar code snippets from the open codebase as few-shot context for code completion
- **Text-to-SQL systems** (Salesforce DAIL-SQL, DIN-SQL): Dynamic example retrieval boosts SQL accuracy by 5–15% over static few-shot
- **Customer support bots**: Retrieve past solved tickets as examples for consistent tone and resolution format
- **Medical coding**: Retrieve similar clinical notes with correct ICD codes as examples for coding new notes
- **Structured extraction**: Retrieve examples of correctly-formatted JSON extractions to guide schema-constrained generation
