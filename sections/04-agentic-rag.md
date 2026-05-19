# 04 — Agentic RAG

> An LLM agent decides *when*, *what*, and *how* to retrieve — issuing multiple queries, using tools, and iterating.

---

## Q1. What is Agentic RAG and how does it differ from pipeline-based RAG? `[Basic]`

**Answer:**

In pipeline RAG, retrieval happens exactly once in a fixed sequence: query → retrieve → generate.

In **Agentic RAG**, an LLM agent controls the retrieval loop:

- It **decides whether to retrieve** at all (some questions don't need it)
- It can **issue multiple retrieval calls** with different queries
- It can **use tools** beyond vector search (web, SQL, APIs)
- It can **iterate** — retrieve, read the results, decide to retrieve more if needed

This is powered by frameworks like **ReAct** (Reasoning + Acting), where the LLM alternates between Thought → Action → Observation steps until it has enough information to answer.

---

## Q2. Explain the ReAct pattern and how it enables agentic retrieval. `[Intermediate]`

**Answer:**

**ReAct** (Yao et al., 2022) interleaves reasoning traces with actions:

```
Thought: The user asked about Q3 revenue. I should look up the financial report.
Action: search("Q3 2024 revenue financial report")
Observation: [Retrieved chunk: "Q3 revenue was $4.2B, up 12% YoY..."]
Thought: I have the revenue figure. Now I need the YoY comparison context.
Action: search("Q3 2023 revenue comparison")
Observation: [Retrieved chunk: "Q3 2023 revenue was $3.75B..."]
Thought: I have enough to answer.
Final Answer: Q3 2024 revenue was $4.2B, a 12% increase from Q3 2023's $3.75B.
```

This enables **multi-hop retrieval** — following a chain of evidence — which single-shot RAG cannot do.

---

## Q3. What is FLARE and how does it improve on ReAct for RAG? `[Intermediate]`

**Answer:**

**FLARE (Forward-Looking Active Retrieval)** is a technique where the model retrieves *proactively* — it predicts what it's about to say and retrieves if it's uncertain:

1. The LLM begins generating a response token by token.
2. When token probability falls below a threshold (model is uncertain), it **pauses**.
3. It uses the partial generation as a query to retrieve supporting context.
4. It resumes generation with the new context.

**Advantage over ReAct:** FLARE doesn't require the LLM to explicitly plan tool use — uncertainty itself triggers retrieval, making it more natural and less prompt-engineered.

**Limitation:** Requires access to token-level probabilities, which isn't available from all LLM APIs.

---

## Q4. What are the risks of Agentic RAG and how do you mitigate them? `[Advanced]`

**Answer:**

| Risk | Description | Mitigation |
|---|---|---|
| **Infinite loops** | Agent keeps retrieving without converging | Set max iteration limits |
| **Prompt injection** | Malicious content in retrieved docs hijacks the agent | Sanitize retrieved content, use system-level guardrails |
| **Runaway costs** | Many LLM + retrieval calls per query | Budget caps, fallback to simple RAG above a cost threshold |
| **Hallucinated tool calls** | Agent invents tool names or parameters | Constrain tool schemas, validate outputs |
| **Latency** | Multi-step loops add seconds per query | Set timeouts, cache intermediate retrievals |

Testing agentic systems requires **trace-level evaluation** (not just final answer quality) — tools like LangSmith and Arize Phoenix help here.

---

## Q5. How would you design an Agentic RAG system for a customer support use case? `[Advanced]`

**Answer:**

**Architecture:**

```
User Query
    │
    ▼
Intent Classifier (LLM)
    ├── "order status"   → SQL tool (orders DB)
    ├── "product info"   → Vector store (product docs)
    ├── "return policy"  → Vector store (policy docs)
    └── "complex issue"  → Multi-step ReAct agent
                              ├── Tool: CRM lookup
                              ├── Tool: Order history
                              └── Tool: Knowledge base search
```

**Key design decisions:**

1. **Fast path for simple queries** — route common intents directly to tools (no agent loop needed).
2. **Agent only for complex queries** — reduces latency and cost for the majority of traffic.
3. **Guardrails** — always end with a human-escalation option if agent confidence is low.
4. **Audit trail** — log every tool call and observation for compliance and debugging.
5. **Fallback** — if the agent exceeds 5 iterations, escalate to a human agent.
