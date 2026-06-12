# 07 — Conversational Context Drift

> In multi-turn RAG, conversation history accumulates pronouns, implicit references, and topic shifts that corrupt the retrieval query — causing the system to retrieve context for the wrong topic.

---

## Q1. What is conversational context drift and how does it differ from single-turn retrieval failure? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Conversational context drift** occurs in multi-turn RAG systems when earlier turns in the conversation cause the retrieval query for the current turn to be incorrect or misleading.

**Single-turn retrieval failure:** The query itself is bad (ambiguous, wrong vocabulary, etc.) — the problem exists in the query independent of conversation.

**Conversational context drift:** The current turn's query is reasonable in isolation, but when interpreted with the conversation history, it generates an incorrect or misleading retrieval signal.

**Example:**

```
Turn 1: "Tell me about our data retention policy."
System: [Retrieves and answers from data retention policy document]

Turn 2: "How long does it apply to employee records specifically?"
```

Naively appending Turn 2 to the retrieval query → "How long does it apply to employee records specifically?" — this fragment, without context, is ambiguous. But if the RAG system constructs the retrieval query from only Turn 2, it may retrieve generic HR retention documents rather than the data retention policy already discussed.

**The failure has two forms:**

1. **Pronoun/reference failure** — "How long does *it* apply" — the pronoun "it" refers to the data retention policy from Turn 1. The retriever doesn't know what "it" refers to.

2. **Topic drift** — The user shifts topic mid-conversation without an explicit signal. The retriever, still conditioned on earlier context, retrieves documents about the old topic.

</details>

---

## Q2. What are the root causes of conversational context drift? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Root Cause | Description | Example |
|---|---|---|
| **Coreference without resolution** | Pronouns ("it", "they", "this") refer to entities from earlier turns; retrieval query contains unresolved pronouns | "What does it say about exceptions?" when "it" = policy document from Turn 3 |
| **Implicit topic continuation** | User assumes context; asks follow-up without repeating the topic | "Can I opt out?" after discussing cookie consent (but retriever doesn't know the topic) |
| **Topic shift without signal** | User switches topics mid-conversation; system retrieves for old topic + new query | "Actually, how does this compare to GDPR?" after discussion of internal policy |
| **Conversational ellipsis** | User omits subject ("What about Canada?") because it's obvious from context | Retriever gets "What about Canada?" with no subject |
| **History poisoning** | Earlier turns' answers contaminate the embedding of the current query | User asked about Product A for 5 turns; Turn 6 asks about Product B but embedding is pulled toward A |
| **Temporal reference** | "Last time you mentioned X" — X is from a prior session, not the current context | System retrieves for X but cannot find it in the current history |

**Most dangerous combination:** Coreference + topic shift in the same turn. The user switches topics AND uses implicit references: "What does that one say about refunds?" — "that one" could be a different document from Turn 2, and "refunds" is a new subtopic.

</details>

---

## Q3. How do you detect conversational context drift before it causes a retrieval failure? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Detection strategy 1 — Standalone query coherence check**

Test whether the current turn's query makes sense without conversation history:

```python
def is_standalone_query(current_turn: str, threshold: float = 0.6) -> bool:
    # Measure how much the query depends on context
    # Simple heuristic: check for pronouns and demonstratives
    reference_signals = ["it", "this", "that", "they", "them", "those",
                         "the above", "as mentioned", "what you said",
                         "the previous", "the same"]
    has_references = any(signal in current_turn.lower()
                         for signal in reference_signals)
    # Also check if query starts with continuation words
    continuation_starts = ["and ", "but ", "also ", "what about ", "how about "]
    is_continuation = any(current_turn.lower().startswith(s)
                          for s in continuation_starts)
    return not (has_references or is_continuation)
```

**Detection strategy 2 — Embedding-based continuity score**

Embed the standalone current turn and the contextualized (history-appended) version. If they are far apart in embedding space, the turn relies heavily on context:

```python
def context_dependence_score(history: str, current_turn: str) -> float:
    standalone_emb = embed_model.embed(current_turn)
    contextualized = f"{history}\n{current_turn}"
    contextualized_emb = embed_model.embed(contextualized)
    # Low cosine similarity = high context dependence
    return 1 - cosine_similarity(standalone_emb, contextualized_emb)
    # Score > 0.3 suggests high context dependence
```

**Detection strategy 3 — LLM-based intent classifier**

Prompt a fast LLM to classify whether the turn can be answered standalone:
```
"Does the following question require context from previous conversation to be understood?
Question: {current_turn}
Answer: yes/no"
```

</details>

---

## Q4. What is query condensation and how does it solve context drift? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Query condensation** (also called query contextualization or history-aware query rewriting) rewrites the current turn into a standalone query that encodes all necessary context from the conversation history.

**Before condensation:**
```
History:
  User: "Tell me about the data retention policy."
  Assistant: "Our data retention policy specifies that customer data is kept for 7 years..."

Current turn (context-dependent):
  "How long does it apply to employee records specifically?"
```

**After condensation:**
```
Standalone query:
  "How long does the data retention policy apply to employee records?"
```

**Implementation:**

```python
CONDENSATION_PROMPT = """Given the following conversation history and the latest user message,
rewrite the user message as a standalone question that contains all necessary context.
If the message is already standalone, return it unchanged.

Conversation History:
{history}

Latest Message: {current_turn}

Standalone question (one sentence, no preamble):"""

def condense_query(history: list[dict], current_turn: str) -> str:
    history_text = "\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}"
        for turn in history[-6:]  # Last 3 turns (6 messages)
    )
    condensed = llm.invoke(CONDENSATION_PROMPT.format(
        history=history_text,
        current_turn=current_turn
    ))
    return condensed.strip()
```

**Key design decision:** How much history to include in the condensation prompt?
- Too little: misses important context (coreference fails)
- Too much: increases cost; stale context from early turns may poison condensation

**Recommendation:** Include the last 3–5 turns (6–10 messages). Earlier turns rarely affect retrieval relevance.

</details>

---

## Q5. How does LangChain implement conversational RAG and what are its limitations? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

LangChain provides `create_history_aware_retriever` for conversational RAG:

```python
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Query condensation chain
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question "
               "which might reference context in the chat history, "
               "formulate a standalone question. Do NOT answer the question, "
               "just reformulate it if needed, otherwise return as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Full conversational RAG chain
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant. Use the following context:\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Usage
response = rag_chain.invoke({
    "input": "How long does it apply to employee records?",
    "chat_history": chat_history  # List of HumanMessage/AIMessage
})
```

**LangChain's limitations:**

1. **History truncation not automatic** — LangChain does not automatically truncate long conversation histories. For long conversations, the history passed to condensation may exceed the context window.

2. **Condensation quality depends on the LLM** — Weak LLMs produce poor condensations (still context-dependent or hallucinate references).

3. **No drift detection** — LangChain does not detect when condensation is needed vs. when the turn is already standalone (always runs condensation, wasting LLM calls on standalone turns).

4. **No topic shift handling** — If the user explicitly changes topics ("Let's talk about something else"), the history-aware retriever may still incorporate old-topic context into the new query.

</details>

---

## Q6. How does topic shift differ from topic continuation, and how do you handle each? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| | Topic Continuation | Topic Shift |
|---|---|---|
| **User behavior** | Follow-up question on the same subject | Changes to a new, unrelated subject |
| **Signal** | References existing entities ("it", "the policy", "that document") | New entities, pivot words ("actually", "different question", "what about") |
| **Correct behavior** | Condense query including previous context | Start fresh — ignore previous topic context |
| **Failure if wrong** | Context drift (missing references) | Context poisoning (old topic contaminates new retrieval) |

**Detecting topic shift:**

```python
SHIFT_DETECTION_PROMPT = """Does the user's latest message represent:
A) A continuation of the current topic
B) A shift to a new, unrelated topic

Conversation topic so far: {topic_summary}
Latest message: {current_turn}
Answer A or B only:"""

def detect_topic_shift(history: list[dict], current_turn: str) -> bool:
    topic_summary = summarize_topic(history)
    result = fast_llm.invoke(SHIFT_DETECTION_PROMPT.format(
        topic_summary=topic_summary,
        current_turn=current_turn
    ))
    return result.strip().upper().startswith("B")
```

**Handling each case:**

```python
def build_retrieval_query(history, current_turn):
    if detect_topic_shift(history, current_turn):
        # Topic shift: use current turn as-is, ignore history
        return current_turn
    
    elif not is_standalone_query(current_turn):
        # Topic continuation with references: condense
        return condense_query(history, current_turn)
    
    else:
        # Already standalone: use directly
        return current_turn
```

**Edge case — partial topic shift:** "We've been talking about data retention, but what about GDPR specifically?" — this is both a continuation (data retention) and a shift (new regulation). Handle by condensing with the full context: "How does GDPR relate to our data retention policy?"

</details>

---

## Q7. How do you handle very long conversation histories that exceed the context window? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Long conversations create two context window problems:
1. The condensation LLM prompt exceeds the window (can't include full history for condensation).
2. The generation LLM prompt exceeds the window (can't include full history + retrieved context + current turn).

**Strategy 1 — Sliding window history**

Keep only the last N turns in the active history. N=5–10 is sufficient for most conversational coherence:
```python
MAX_HISTORY_TURNS = 5
active_history = full_history[-MAX_HISTORY_TURNS * 2:]  # 2 messages per turn
```

Risk: if the user references something from Turn 1 in Turn 20, that reference can't be resolved.

**Strategy 2 — Conversation summary memory**

Maintain a running summary of the conversation:
```python
SUMMARY_PROMPT = """Progressively summarize the conversation below,
adding to the previous summary. Return a new summary.

Previous summary: {summary}
New conversation: {new_turns}
New summary:"""

class ConversationSummaryMemory:
    def __init__(self):
        self.summary = ""
        self.recent_turns = []  # Last 3 turns verbatim
    
    def add_turn(self, human_msg, ai_msg):
        self.recent_turns.append((human_msg, ai_msg))
        if len(self.recent_turns) > 3:
            oldest = self.recent_turns.pop(0)
            self.summary = llm.invoke(SUMMARY_PROMPT.format(
                summary=self.summary,
                new_turns=f"Human: {oldest[0]}\nAI: {oldest[1]}"
            ))
    
    def get_context_for_condensation(self):
        return f"Summary so far: {self.summary}\n\nRecent turns: {self.recent_turns}"
```

**Strategy 3 — Entity memory**

Track which entities (documents, policies, products) have been discussed. For condensation, provide only the relevant entity mentions:
```python
entity_memory = {
    "data_retention_policy": "discussed in turns 1-3; 7-year retention, applies to all PII"
}
# Condensation can reference: "data retention policy (7-year retention, PII)"
```

</details>

---

## Q8. How does conversational context drift interact with multi-tenant access control? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

In multi-tenant systems, conversational context drift creates a specific access control risk:

**The cross-tenant leak scenario:**

Consider a multi-tenant system where User A (Tenant A) and User B (Tenant B) use the same underlying RAG system but have isolated document corpora.

User A has a conversation about "Project Aurora" (a Tenant A confidential project). The conversation is correctly isolated to Tenant A's corpus.

In a poorly implemented system where conversation history is managed globally (or if session IDs are misconfigured), User B's new query might accidentally include User A's conversation history as context for condensation — leaking that "Project Aurora" exists.

**Mitigations:**

1. **Strict session isolation** — Conversation histories must be stored per-user-session, never shared across users or tenants. Each session ID must be cryptographically tied to the user's identity.

2. **History purging on session end** — Don't persist conversation histories beyond the session unless explicitly required. Minimize the window of potential cross-session leakage.

3. **Access control on condensed queries** — After condensation, the condensed standalone query may contain entity names or terms that were only accessible via the conversation history. Before using the condensed query for retrieval, verify that the user still has access to all referenced entities (late-binding ACL check).

**The history injection attack:**

An adversarial user may deliberately reference non-existent topics to probe the system: "You mentioned earlier that [false claim]." If the condensation LLM is not robust to this, it may incorporate the false claim into the condensed query — retrieving documents the user wouldn't have been able to find with a direct query.

**Mitigation:** Constrain condensation to only reference entities explicitly present in the stored conversation history.

</details>

---

## Q9. How do you measure conversational RAG quality in production? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Conversational RAG introduces metrics beyond single-turn RAG:

**Retrieval metrics:**

| Metric | How to measure |
|---|---|
| **Context drift rate** | % of multi-turn queries where retrieved context is from the wrong topic |
| **Reference resolution accuracy** | % of pronoun/reference queries where condensed query correctly resolves the reference |
| **Condensation quality** | Semantic similarity between condensed query and the ideal standalone query (human-labeled on 100-query sample) |
| **Topic shift detection accuracy** | % of topic shifts correctly identified and treated as fresh queries |

**End-to-end quality:**

```python
# Multi-turn eval set: conversation transcripts with expected answers per turn
MULTI_TURN_EVAL = [
    {
        "conversation": [
            {"role": "user", "content": "Tell me about our refund policy."},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "What's the deadline?"},  # Turn under test
        ],
        "expected_retrieved_doc": "refund_policy_doc",
        "expected_answer_contains": ["30 days", "deadline"]
    },
    ...
]
```

**Online metrics (production signals):**

| Signal | What it indicates |
|---|---|
| **Follow-up rate** | User immediately asks a clarifying question after an answer → answer didn't address the right topic |
| **Topic restart rate** | User explicitly restates the full question (not just a follow-up) → context drift detected by user |
| **Session abandonment after turn 3+** | Long conversations failing → context drift worsens with turns |
| **Rephrase rate on turn N>1** | User rephrases same question → context drift caused wrong retrieval |

**Drift detection A/B test:**

Compare conversational RAG with vs. without query condensation on a multi-turn query set. Measure Recall@5 on turns 3+ of the conversation — the improvement from condensation is largest on later turns where context has accumulated.

</details>

---

## Q10. What are the best practices for multi-turn RAG system design to minimize context drift? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Architecture best practices:**

```
MULTI-TURN RAG SYSTEM (context-drift-resistant)

State management:
  ├─ session_id → conversation history (last 10 turns, verbatim)
  ├─ session_id → conversation summary (running LLM-generated summary)
  ├─ session_id → active entities {entity_name: brief_description}
  └─ session_id → current topic (periodically updated)

Per-turn pipeline:
  1. Classify turn type:
       a. Standalone query → no condensation needed
       b. Context-dependent continuation → condense with recent history
       c. Topic shift → use current turn only, reset active entities
  
  2. Condense if needed:
       Input: conversation summary + last 3 turns verbatim + current turn
       Output: standalone query
       Quality gate: condensed query must be > 0.7 cosine similarity
                     to current turn (to detect hallucinated references)
  
  3. Retrieve:
       Use condensed/original query
       Apply user's access control filter
  
  4. Generate:
       Include: system prompt + conversation summary + last 3 turns +
                retrieved context + current turn
       Do NOT include: full raw history (use summary instead)
  
  5. Update state:
       Append new turn to history
       Update summary (if history > 3 turns)
       Update active entities
       Update topic estimate
```

**Anti-patterns to avoid:**

1. **Always including full raw history in the retrieval prompt** — History grows unboundedly; retrieval degrades over time.

2. **Using the full history for condensation** — Stale early turns pollute the condensed query.

3. **Never condensing** — Context-dependent turns produce bad retrieval queries.

4. **Always condensing** — Standalone turns waste an LLM call and introduce condensation errors.

5. **Not resetting state on topic shift** — Old topic context poisons new-topic retrieval.

</details>
