# 43 — Deep Research / Agentic Research RAG

> An orchestrator plans a research project, dispatches multiple search-read-synthesize loops (often across parallel sub-agents), and assembles the results into a long-form, multi-source cited report — trading single-turn latency for report-scale depth.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
Research Brief (user's question/topic)
    │
    ▼
Lead/Orchestrator Agent ──► decomposes into sub-questions, builds a research plan
    │
    ▼
Sub-Agent Dispatch (parallel workers, one per sub-question)
    │
    ├──► Sub-Agent 1: search → read pages → extract findings ──┐
    ├──► Sub-Agent 2: search → read pages → extract findings ──┤
    └──► Sub-Agent N: search → read pages → extract findings ──┘
    │
    ▼
Findings Aggregator (merges sub-agent outputs, dedupes sources, tracks citations)
    │
    ▼
Cost/Latency Budget Monitor ──► loop back to dispatch more sub-agents if budget remains
    │                            and coverage gaps are detected
    ▼
Report Synthesizer (LLM writes long-form report, section by section, with inline citations)
    │
    ▼
Citation Aggregation & Formatting (dozens of sources → numbered bibliography)
    │
    ▼
Final Multi-Page Cited Report
```

### Key Components

| Component | Responsibility |
|---|---|
| Lead/Orchestrator Agent | Turns the research brief into a plan, decomposes it into parallelizable sub-questions |
| Research Sub-Agents | Each investigates one sub-question independently: search → fetch → read → extract findings |
| Findings Aggregator | Merges sub-agent outputs, deduplicates overlapping sources, tracks provenance per finding |
| Budget Monitor | Tracks token/dollar/time spend against a report budget; decides whether to spawn more sub-agents or wrap up |
| Report Synthesizer | Produces the final long-form report, weaving findings into sections with inline citations |
| Citation Aggregator | Collects citations from every sub-agent into a single deduplicated, numbered bibliography |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Commercial products | OpenAI Deep Research (ChatGPT), Gemini Deep Research (Google), Perplexity Labs |
| Open-source frameworks | LangChain `open_deep_research`, Hugging Face smolagents `open_deep_research` |
| Multi-agent orchestration | LangGraph (supervisor/worker graphs), CrewAI, AutoGen |
| Search/fetch backend | Tavily, Bing Search API, native web search tool use (Anthropic/OpenAI) |
| Evaluation | GAIA benchmark (general assistant tasks), DeepSearchQA (Google) |

---

## Q1. What is Deep Research and how does it differ from Agentic Web RAG? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Deep Research** is an agentic pattern that produces a long-form, multi-source, cited *report* on a topic by running many search→read→synthesize cycles — often through parallel sub-agents — over a budget of minutes (sometimes tens of minutes) and dollars, rather than a single request-response turn. OpenAI's Deep Research (launched Feb 2025) and Gemini Deep Research are the flagship commercial examples; open-source implementations include LangChain's `open_deep_research` and Hugging Face smolagents' `open_deep_research`.

**Agentic Web RAG** (file 31, Perplexity-style) is the single-turn sibling: one query planner, one or a few web searches, one synthesized answer with inline citations — designed to return in seconds.

```
Agentic Web RAG (file 31):
  "What's the current inflation rate in the US?"
  → 1-3 searches → fetch top pages → single-paragraph cited answer
  Latency: seconds. Sources: ~3-10. Output: a paragraph.

Deep Research (this file):
  "Analyze the competitive landscape of GLP-1 weight-loss drugs and
   project market share shifts over the next 3 years."
  → decompose into ~5-10 sub-questions (market size, competitors, pipeline
    drugs, regulatory landscape, pricing trends...)
  → dispatch parallel sub-agents, each running its OWN multi-step search loop
  → aggregate findings, resolve conflicting sources, cite everything
  → synthesize a multi-page report with a bibliography
  Latency: minutes (5-30+). Sources: dozens. Output: a structured report.
```

| Dimension | Agentic Web RAG (file 31) | Deep Research |
|---|---|---|
| Output shape | Single answer/paragraph | Multi-section long-form report |
| Time budget | Seconds | Minutes to tens of minutes |
| Cost per query | Cents | Can run into dollars per report |
| Search depth | 1-3 searches, single agent | Dozens of searches across many sub-agents |
| Orchestration | Single query planner | Lead agent + parallel research sub-agents |
| Best for | Quick factual/current-events lookups | Market research, literature reviews, due diligence |

**Key insight:** Deep Research is not "Agentic Web RAG but with more searches" bolted onto the same loop — it introduces a *planning and decomposition layer* plus *sub-agent parallelism* that Agentic Web RAG's single-turn design doesn't need.

</details>

---

## Q2. How does the sub-agent / worker orchestration pattern work? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A **lead (orchestrator) agent** decomposes the research brief into independent sub-questions, then spawns one **worker sub-agent per sub-question**, running them in parallel to fit within a wall-clock time budget. Each worker runs its own inner search→read→extract loop, similar in shape to file 04's Agentic RAG loop, but scoped to a narrow sub-question.

```python
from concurrent.futures import ThreadPoolExecutor

def lead_agent_plan(research_brief: str, llm) -> list[str]:
    """Decompose the brief into independent, parallelizable sub-questions."""
    prompt = f"""Break this research task into 4-8 independent sub-questions
that can be researched in parallel. Each should be answerable without
needing the answer to another sub-question.

Research brief: {research_brief}

Return a JSON list of sub-questions."""
    return llm.generate_json(prompt)

def research_subagent(sub_question: str, search_tool, llm, max_steps: int = 6) -> dict:
    """One worker's own search -> read -> extract loop, scoped to one sub-question."""
    findings, sources = [], []
    query = sub_question
    for step in range(max_steps):
        results = search_tool.search(query, k=5)
        for r in results:
            page_text = fetch_and_clean(r["url"])
            extracted = llm.generate(
                f"Extract facts relevant to '{sub_question}' from:\n{page_text}"
            )
            findings.append(extracted)
            sources.append(r["url"])
        # Sub-agent decides whether it has enough or needs a refined query
        next_query = llm.generate(
            f"Given findings so far: {findings}\nDo you need another search? "
            f"If so, what query? If not, respond DONE."
        )
        if next_query.strip() == "DONE":
            break
        query = next_query
    return {"sub_question": sub_question, "findings": findings, "sources": sources}

def deep_research(research_brief: str, search_tool, llm) -> dict:
    sub_questions = lead_agent_plan(research_brief, llm)
    with ThreadPoolExecutor(max_workers=len(sub_questions)) as pool:
        results = list(pool.map(
            lambda q: research_subagent(q, search_tool, llm), sub_questions
        ))
    return {"sub_question_results": results}
```

**Why parallelize across sub-agents rather than one long sequential loop?**

- Wall-clock latency: 8 sub-questions run concurrently in ~1 sub-agent's time budget instead of 8x that time sequentially
- Context isolation: each sub-agent's context window stays focused on one sub-question, avoiding the "lost in the middle" degradation of stuffing everything into one giant agentic loop (relevant to file 10, Long-Context RAG, tradeoffs)
- Failure isolation: one sub-agent hitting a dead end (paywalled sources, no results) doesn't block the others

</details>

---

## Q3. How is the final report synthesized and how are citations aggregated across dozens of sources? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

After sub-agents return, the orchestrator must merge findings that may be redundant, contradictory, or differently sourced, then write a coherent report where every claim still traces back to a specific source.

```python
def aggregate_citations(sub_agent_results: list[dict]) -> dict:
    """Dedupe sources across all sub-agents, assign a single global citation ID."""
    url_to_id = {}
    next_id = 1
    for result in sub_agent_results:
        for url in result["sources"]:
            if url not in url_to_id:
                url_to_id[url] = next_id
                next_id += 1
    return url_to_id

def synthesize_report(research_brief: str, sub_agent_results: list[dict], llm) -> str:
    citation_map = aggregate_citations(sub_agent_results)

    findings_block = ""
    for result in sub_agent_results:
        findings_block += f"\n## {result['sub_question']}\n"
        for finding, src in zip(result["findings"], result["sources"]):
            cid = citation_map[src]
            findings_block += f"- {finding} [{cid}]\n"

    bibliography = "\n".join(
        f"[{cid}] {url}" for url, cid in sorted(citation_map.items(), key=lambda x: x[1])
    )

    prompt = f"""Write a structured, multi-section report answering this research
brief, using ONLY the findings below. Preserve every citation marker [N] next
to the claim it supports. Resolve any contradicting findings by noting the
disagreement explicitly rather than silently picking one.

Research brief: {research_brief}

Findings (grouped by sub-question):
{findings_block}
"""
    report_body = llm.generate(prompt, max_tokens=4000)
    return f"{report_body}\n\n## Sources\n{bibliography}"
```

**Handling contradictions across sub-agents:** unlike single-turn RAG where one retrieval set is used once, Deep Research routinely surfaces conflicting numbers from different sources (e.g. two market-size estimates that differ by 2x). The synthesizer is explicitly prompted to surface disagreement ("Source A estimates X; Source B estimates Y") rather than silently averaging or picking one — this is a distinct failure mode from standard RAG's single-hop citation problem (file 33, Verifiable/Citation RAG) because the *inputs themselves* disagree, not just the model's grounding of them.

</details>

---

## Q4. How do you budget cost and latency across dozens of searches per report? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A Deep Research run can spawn dozens of searches, page fetches, and LLM calls — each sub-agent step costs tokens (LLM calls) and possibly a paid search API call. Left unbounded, a single report can cost several dollars and take much longer than a user will tolerate. Production systems budget explicitly:

```python
class ResearchBudget:
    def __init__(self, max_dollars: float = 2.00, max_seconds: int = 900,
                 max_subagents: int = 8, max_searches_per_subagent: int = 6):
        self.max_dollars = max_dollars
        self.max_seconds = max_seconds
        self.max_subagents = max_subagents
        self.max_searches_per_subagent = max_searches_per_subagent
        self.spent_dollars = 0.0
        self.start_time = time.time()

    def can_continue(self) -> bool:
        elapsed = time.time() - self.start_time
        return self.spent_dollars < self.max_dollars and elapsed < self.max_seconds

    def charge(self, llm_call_cost: float):
        self.spent_dollars += llm_call_cost
```

**Budget levers reported by real products:**

| Lever | Example |
|---|---|
| Query tiers | OpenAI ChatGPT Pro: 250 deep-research queries/month, half "lightweight" (cheaper, faster, shallower); Plus/Team: 25/month |
| Time cap | OpenAI Deep Research browses for roughly 5-30 minutes per report |
| Sub-agent cap | Limit the lead agent's plan to N sub-questions regardless of how many it would ideally want |
| Early termination | If the budget monitor detects the marginal new finding rate has dropped (most new searches return already-seen facts), stop dispatching new sub-agents even if budget remains |
| Model tiering | Use a cheaper/faster model for sub-agent search-and-extract steps, reserve the most capable model for final synthesis |

**The core trade-off:** more sub-agents and more searches generally improve coverage and reduce the risk of missing a key source, but cost and latency scale roughly linearly with sub-agent count — production systems must cap this well before "complete" coverage, accepting some recall loss for bounded cost.

</details>

---

## Q5. What failure modes are unique to Deep Research, and how would you combine it with other bank architectures to mitigate them? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Failure mode 1 — citation dilution across dozens of sources.** With single-turn RAG (file 33), verifying 3-10 citations against their source passages is tractable. At report scale with 30-80 sources, exhaustive per-claim NLI verification becomes expensive and slow — teams often verify only a sample or rely on the synthesizer's self-reported citation placement, which reintroduces the "hallucinated citation" risk that file 33 was built to solve. Mitigation: run file 33's attribution-verification step, but only on a report's *load-bearing claims* (numbers, direct quotes, contested statements) rather than every sentence, to keep verification cost bounded.

**Failure mode 2 — redundant/wasted sub-agent work.** Sub-agents dispatched in parallel don't see each other's findings mid-flight, so two sub-agents can independently research overlapping territory, burning budget without adding coverage. Mitigation: a lighter-weight coordination pass (sub-agents periodically report a one-line status back to the lead agent, which can redirect an idle sub-agent to an uncovered angle) — this pushes Deep Research toward the same dynamic re-planning pattern used in Adaptive RAG (file 11), but applied at the sub-agent-plan level instead of the single-query level.

**Failure mode 3 — stale or low-quality source over-reliance.** Because Deep Research optimizes for coverage within a budget, sub-agents under time pressure may accept the first few search results rather than critically filtering for authoritative sources, especially for fast-moving topics. Mitigation: add a lightweight source-quality filter (domain reputation, publication date recency) as a gate before a sub-agent's findings are handed to the aggregator — analogous to the freshness handling in Streaming/Real-Time RAG (file 35).

**Combining with other architectures:**

- **+ Verifiable Citation RAG (file 33):** sampled attribution verification on load-bearing claims only, to keep report-scale citation checking affordable
- **+ Adaptive RAG (file 11):** dynamic re-planning of the sub-question set mid-run based on early sub-agent findings, instead of a static upfront plan
- **+ Search-R1-style RL search (file 42):** replace each sub-agent's prompted search loop with an RL-trained search policy, reducing wasted/redundant searches per sub-agent since the retrieval decisions are learned rather than heuristically prompted

</details>

---

## Q6. Walk through the Deep Research architecture end-to-end, from a research brief to a final report. `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

```
Research Brief
    │
    ▼
Lead/Orchestrator Agent ──► decomposes into sub-questions, builds a research plan
    │
    ▼
Sub-Agent Dispatch (parallel workers, one per sub-question)
    ├──► Sub-Agent 1: search → read pages → extract findings ──┐
    ├──► Sub-Agent 2: search → read pages → extract findings ──┤
    └──► Sub-Agent N: search → read pages → extract findings ──┘
    │
    ▼
Findings Aggregator (merges outputs, dedupes sources, tracks citations)
    │
    ▼
Budget Monitor ──► loop back to dispatch more sub-agents if budget remains
    │               and coverage gaps are detected
    ▼
Report Synthesizer (writes long-form report, section by section, with citations)
    │
    ▼
Citation Aggregation & Formatting → Final Multi-Page Cited Report
```

Each stage exists to solve a problem single-turn RAG doesn't have to: the lead agent's decomposition step exists because no single search-and-synthesize pass can cover a multi-faceted research brief; parallel dispatch exists to fit dozens of searches into a wall-clock time budget rather than running them sequentially; the aggregator exists because findings from independent sub-agents will overlap and sometimes conflict (Q3); and the budget monitor exists because, unlike a single-turn system with an implicit one-shot cost, this architecture's cost and latency are open-ended unless explicitly capped (Q4).

</details>

---

## Q7. What is the single distinctive mechanism that separates Deep Research from a single agentic loop with more search calls? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The distinctive mechanism is **upfront decomposition into independently-parallelizable sub-questions**, not simply "run more search rounds in one loop." A single agentic loop with a higher `max_rounds` (as in Agentic Web RAG's multi-step research, #31 Q5) still processes searches sequentially within one shared context, accumulating everything into one growing conversation history — which is exactly the "lost in the middle" risk Q2 and Q14 describe. Deep Research instead splits the *problem itself* into sub-questions each narrow enough for a dedicated sub-agent to research in its own isolated context, then merges the independently-produced findings afterward.

This single structural choice is what enables both of Deep Research's other defining properties: parallelism (independent sub-agents can run concurrently, since decomposition made them non-interdependent) and depth (each sub-agent's context stays focused on one facet rather than diluting across the whole research brief). Simply increasing a single loop's round count would still hit the same lost-in-the-middle ceiling that motivated the decomposition in the first place.

</details>

---

## Q8. What is the practical origin of Deep Research as a product pattern? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

OpenAI's Deep Research (launched within ChatGPT in February 2025) is generally credited as the product that established this pattern at scale — autonomously browsing the web for 5-30 minutes to produce analyst-level, cited reports. Gemini Deep Research followed a similar architecture from Google, and open-source replication efforts (LangChain's `open_deep_research`, Hugging Face smolagents' `open_deep_research`) emerged shortly after, benchmarked against the GAIA general-assistant-task benchmark to measure how closely an open implementation could match the commercial products' capability.

Like Agentic Web RAG (#31, Q8), Deep Research doesn't trace to a single academic paper — it's a product-driven architectural pattern that emerged once multi-agent orchestration frameworks (LangGraph, CrewAI, AutoGen) and reliable long-running tool-use agents matured enough to make minutes-long, dozens-of-searches research sessions practical and affordable at consumer scale.

</details>

---

## Q9. How does Deep Research compare to CoRAG (#50) and other iterative-retrieval architectures? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

CoRAG (#50) and similar iterative/multi-hop retrieval architectures (#19) chain a *sequence* of retrieval steps within a single reasoning path, where each retrieval informs the next step of the same chain — the retrieval is deep in the sense of "many sequential hops," but the output is still typically a single, focused answer to a single question. Deep Research is deep in a different dimension: it's *wide* (parallel sub-agents covering independent facets of a broader topic) as much as it is deep, and its output is a structured, multi-section report rather than a single answer.

The practical distinction: use CoRAG-style iterative retrieval when a question genuinely requires chaining several retrieval steps to reach one answer ("what is the birthplace of the director of the highest-grossing film of 2019"); use Deep Research when the request is inherently multi-faceted and the user wants a report covering several independent angles ("analyze the competitive landscape of X"). Deep Research's sub-agents can themselves use iterative retrieval internally (Q2's `research_subagent` loop), making the two patterns complementary rather than competing — CoRAG-style chaining is a reasonable choice for what happens *inside* one sub-agent, while Deep Research's decomposition operates one level above that.

</details>

---

## Q10. What are the key tuning knobs for a Deep Research system, and how do you choose them? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Knob | Effect | Starting point |
|---|---|---|
| Number of sub-questions (lead agent's plan size) | More sub-questions improve coverage but multiply parallel compute and aggregation complexity | 4-8, per Q2's prompt |
| `max_searches_per_subagent` | Bounds how deep each individual sub-agent can go before returning | 6, per Q4's budget class |
| `max_subagents` / `max_dollars` / `max_seconds` (Q4's `ResearchBudget`) | Hard ceilings on total cost and latency regardless of how much the plan would ideally want | Set per product tier (Q4's OpenAI query-tier example: lightweight vs. full-depth runs) |
| Model tiering (cheap model for sub-agent search/extract, strong model for synthesis) | Controls the cost-quality trade-off across the pipeline's most expensive stage (synthesis) vs. its highest-volume stage (sub-agent steps) | Cheap/fast model for the many sub-agent steps; reserve the most capable model for the single, high-stakes synthesis call |

The number of sub-questions and the per-sub-agent search budget interact multiplicatively with total cost (roughly `sub_questions x searches_per_subagent x cost_per_search`), which is why the hard budget caps in Q4 exist as an independent backstop rather than trusting the multiplication of "reasonable-looking" individual knobs to stay within an acceptable total — a lead agent proposing 8 sub-questions each running its full 6-search budget can still produce an unexpectedly expensive report if nothing caps the aggregate.

</details>

---

## Q11. How do you evaluate a Deep Research system's report quality? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Report-scale evaluation needs different tooling than single-answer RAG evaluation, since there's no single "correct answer" string to match against. The GAIA benchmark (general AI assistant tasks) and Google's DeepSearchQA are the standard external references — both designed to measure an agent's ability to autonomously research and synthesize correct, well-supported answers to non-trivial, multi-step questions rather than simple lookups. Hugging Face's open-source replication reported roughly 55% pass@1 on GAIA versus roughly 67% for OpenAI's original Deep Research, illustrating that faithfully reproducing the full pipeline's quality is itself non-trivial even when the architectural pattern is well understood.

For an internal deployment, build a golden set of representative research briefs with a rubric-based evaluation (coverage of expected sub-topics, citation accuracy sampled per Q17 of file 33's approach, internal consistency/contradiction handling per Q3) rather than expecting exact-match scoring — an LLM-as-judge comparing the report against a rubric of "must address these facets, must cite claims of type X" is the practical approach most teams converge on, since human review of every report at development-iteration speed doesn't scale.

</details>

---

## Q12. What is the characteristic failure mode of sub-agent context isolation? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Context isolation (Q2's rationale: each sub-agent stays focused, avoiding lost-in-the-middle) has a direct cost: a sub-agent researching "market size" has no visibility into what the sub-agent researching "regulatory landscape" is finding, even when the two facets are genuinely entangled — a regulatory change discovered by one sub-agent might materially change how the other's market-size estimate should be interpreted, but neither sub-agent, working in isolation, has the context to notice the connection.

**Symptom:** the synthesized report reads as a collection of independently-correct sections that don't cross-reference each other where they should, or worse, contain findings that are individually accurate but jointly misleading once combined (e.g., a market-size sub-agent's estimate that a regulatory sub-agent's findings, had they been visible, would have qualified). **Mitigation:** the lightweight coordination pass described in Q5's failure mode 2 (sub-agents periodically reporting status back to the lead agent) can be extended to surface cross-cutting findings specifically — the lead agent, seeing all sub-agents' interim findings, is positioned to flag "sub-agent 3's regulatory finding may be relevant to sub-agent 1's market analysis" and either inject that cross-reference into the final synthesis prompt or redirect a sub-agent to investigate the connection directly.

</details>

---

## Q13. How do you handle a sub-agent that reaches a dead end with no useful findings? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A sub-question can dead-end for several distinct reasons that call for different responses: the underlying information genuinely doesn't exist publicly (a private company's exact revenue figures), the sub-agent's search queries were poorly formulated for a topic that does have public information, or the topic is paywalled/inaccessible to the fetcher (Q31's page-extraction failure modes apply equally here). Silently returning an empty or thin findings set to the aggregator risks the final report either omitting that facet entirely with no explanation, or the synthesizer papering over the gap with vague, unsupported language.

**Handling:** (1) have the sub-agent explicitly report *why* it couldn't find sufficient information (distinguishing "doesn't exist" from "couldn't locate" from "access-restricted") rather than just returning an empty findings list; (2) surface this explicitly in the final report ("Data on X was not publicly available as of this research") rather than silently dropping the sub-question, since an honest gap is far more useful to the report's reader than a confident-sounding synthesis built on thin evidence; (3) if budget remains (Q4's monitor) and the dead-end looks like a query-formulation problem rather than genuine information scarcity, the lead agent can dispatch a retry with a reformulated query before giving up on that facet entirely.

</details>

---

## Q14. How does the "lost in the middle" trade-off relate to Long-Context RAG (#10), and why does sub-agent isolation avoid it? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Long-Context RAG (#10) addresses "can we fit enough retrieved content into one context window" and runs directly into the well-documented "lost in the middle" phenomenon: LLMs attend less reliably to information placed in the middle of a very long context than to information at the beginning or end, so simply stuffing more retrieved content into one prompt has diminishing (and eventually negative) returns on answer quality, independent of context-window size limits.

A single agentic research loop accumulating dozens of search results into one growing conversation history (Q7's contrast case) runs into exactly this same failure mode — by round 8 of a sequential loop, the model's attention over round 1's findings has degraded relative to round 8's, even though round 1's findings might be equally important to the final answer. Deep Research's sub-agent isolation sidesteps this structurally rather than mitigating it after the fact: each sub-agent's context only ever holds the material relevant to *its own* narrow sub-question, so no single context window ever needs to hold the full research session's accumulated findings — that consolidation only happens once, at the aggregation step (Q3), working from already-distilled findings rather than raw accumulated search history, which is a fundamentally shorter and more attention-friendly input than the single-loop alternative would produce.

</details>

---

## Q15. How would you build a decision-gate benchmark to decide when a query merits a full Deep Research run vs. a single-turn Agentic Web RAG answer? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Given the cost and latency gap between the two (seconds/cents vs. minutes/dollars, Q1's comparison table), routing this decision correctly matters as much as it does for the corpus-vs-web-search routing gate in Agentic Web RAG's own Q16:

```
1. Label a sample of real user queries by whether they're genuinely
   multi-faceted (would benefit from decomposition into independent
   sub-questions) vs. single-focus (answerable well by one search-and-
   synthesize pass) -- query length, presence of multiple distinct
   entities/topics, and explicit "analyze," "compare," "report on"
   language are useful signals.

2. Prototype a router (a cheap classifier or a single upfront LLM call
   asking "would this benefit from decomposition into N independent
   sub-questions") and measure its agreement with the human labels.

3. Route: single-turn Agentic Web RAG for queries the router is
   confident are single-focus; Deep Research for queries flagged as
   multi-faceted, with an explicit user-facing signal ("this will take
   several minutes") given the latency difference is large enough that
   silently routing to the slow path without warning would surprise users.

4. Gate: measure whether the router's Deep-Research-routed queries
   actually show a report-quality improvement over what a single-turn
   answer would have produced (via the same query re-run through both
   paths on a sample) -- justifying the cost multiple, not just
   confirming the query "sounds" complex.
```

Unlike many other routing decisions in this bank, the cost and latency gap here is large enough that most production systems make this an explicit user choice (a "quick answer" vs. "deep research" toggle) rather than a fully automatic router — the stakes of a wrong automatic routing decision (a user waiting 20 minutes for a report when they wanted a quick answer, or vice versa) are higher than for cheaper routing decisions elsewhere in this bank.

</details>

---

## Q16. What is the cost and infrastructure overhead of running Deep Research at scale? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Cost is dominated by the multiplicative structure noted in Q10: `sub_questions x searches_per_subagent x (search API cost + LLM extraction cost per search) + one large synthesis call`. Illustrative worked example at 6 sub-questions, 4 searches each: 24 searches at illustrative $0.01/search plus per-search LLM extraction calls (illustrative $0.002 each on a cheap model) = roughly $0.50 in search/extraction cost, plus a single synthesis call on a strong model processing several thousand tokens of aggregated findings (illustrative $0.10-0.30) — landing in the same "can run into dollars per report" range OpenAI's own product tiers reflect (Q4's 250 lightweight vs. 25 full-depth queries/month distinction).

Infrastructure overhead beyond raw API cost: parallel sub-agent execution requires an orchestration layer capable of running and monitoring several concurrent agent sessions per report (Q2's `ThreadPoolExecutor` is illustrative; production systems typically use a proper job queue or multi-agent framework like LangGraph for this), plus per-report budget tracking (Q4's `ResearchBudget`) that must be enforced across genuinely concurrent workers rather than a single sequential loop, which is a meaningfully more complex operational surface than a single-turn RAG system's request-response model. At scale, this pushes toward product-tier rate limiting (a fixed number of full Deep Research runs per user per period, as OpenAI's tiers illustrate) rather than uniform per-query pricing, since the cost variance between a simple and a maximally-complex research brief is large enough that flat per-query pricing would badly misprice the service.

</details>

---

## Q17. What security and trust risks are specific to a multi-agent research system? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

- **Poisoned sub-agent findings propagating unfiltered into the final report** — since the aggregator (Q3) merges sub-agent outputs largely at face value (deduping sources, but not independently re-verifying every finding), a single sub-agent that encounters and incorporates a poisoned or manipulated source can inject a false claim into the final report with the same apparent authority as every other, correctly-sourced finding — and because the report cites dozens of sources, a single bad one is easy to miss relative to the citation-density scrutiny a single-turn answer with 3-5 citations would get.
- **Prompt injection targeting the decomposition or synthesis step specifically** — a malicious page encountered by one sub-agent could contain content crafted to influence not just that sub-agent's own findings (the general web-RAG injection risk, #31 Q3) but, if that finding text flows into the synthesis prompt largely unfiltered (Q3's `findings_block`), potentially influence the tone or framing of the entire report during the final synthesis pass, a larger blast radius than a single-turn system's injection risk.
- **Resource-exhaustion attacks via adversarial research briefs** — a brief deliberately crafted to maximize legitimate-seeming sub-question count or search depth could push a request toward the upper bound of the cost/time budget (Q4, Q16) repeatedly, which is a more expensive denial-of-service vector than equivalent attacks against a single-turn system given the much higher baseline cost per request.
- **Aggregation-stage source dilution** — with dozens of sources, a report can technically satisfy "every claim has a citation" while a meaningful fraction of those citations are low-quality or unverified (Q5's failure mode 3), which is harder for a reader to spot at report scale than it would be in a short, single-turn answer where each citation gets proportionally more scrutiny.

Mitigation: apply source-quality filtering at the sub-agent level before findings ever reach the aggregator (Q5's mitigation for failure mode 3), sample-based attribution verification weighted toward the report's most load-bearing claims (Q5's mitigation for failure mode 1) rather than assuming aggregate citation density implies aggregate reliability, and treat sub-agent-to-synthesizer findings text with the same prompt-injection wariness applied to any RAG context, scaled up for the larger number of independent content sources involved.

</details>

---

## Q18. Design a Deep Research system for an enterprise competitive-intelligence use case with strict source-reliability requirements. `[Advanced]` `[Scenario]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Requirements:** analysts need multi-page competitive landscape reports; every material claim must be traceable to an authoritative source; the business cost of an undetected bad source (a wrong market-share number driving a real investment decision) is high enough to justify extra verification cost.

```
1. Decomposition (Q2, Q6): lead agent breaks the brief into sub-questions
   aligned to standard competitive-intelligence facets (market size,
   competitor positioning, pricing, regulatory environment, recent
   M&A activity) rather than an unconstrained open-ended plan --
   a domain-specific decomposition template improves both coverage
   consistency and downstream source-quality filtering.

2. Source restriction per sub-agent (extending #31 Q19's domain-
   allowlisting to this multi-agent setting): each sub-agent is
   configured with a tiered source-trust policy specific to its
   facet -- SEC filings and analyst reports for market-size/pricing
   sub-questions, official regulatory sources for the regulatory
   sub-question -- rather than one generic trust policy shared across
   all sub-agents researching very different kinds of claims.

3. Mandatory citation verification (Q17's mitigation) on ALL numeric
   and market-share claims specifically, not just a random sample --
   given this domain's cost of error, the load-bearing-claims sampling
   approach from Q5/Q17 is tightened to 100% coverage for this specific
   claim category while remaining sampled for lower-stakes claims.

4. Cross-sub-agent consistency pass (Q12's mitigation): before final
   synthesis, an explicit pass checks whether market-size and
   competitor-positioning sub-agents' findings are mutually consistent
   (e.g., do individual competitor share estimates sum to a sane total
   market size), flagging discrepancies for human review rather than
   letting the synthesizer silently reconcile them.

5. Human-in-the-loop gate before final delivery: given the stakes,
   the report is routed to an analyst for review of flagged claims
   (low-source-trust findings, cross-agent inconsistencies) before
   being finalized, rather than auto-delivering the synthesizer's
   output directly.
```

The key design choice is tiering verification effort by claim consequence (100% verification on numeric/market-share claims, sampled elsewhere) rather than applying uniform verification depth across the whole report — this keeps the system's cost from scaling with total citation count while still closing the specific gap (Q17) that matters most for this use case's actual risk profile.

</details>

---

## Q19. What happens when the budget monitor's early-termination heuristic misfires, and how do you debug it? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Q4's early-termination heuristic (stop dispatching new sub-agents once the marginal new-finding rate drops) can fail in both directions. **Stopping too early:** if early sub-agents happen to converge on overlapping, easily-discoverable information first (the most obvious search results for related sub-questions often surface similar high-authority sources), the heuristic can mistake this early convergence for "coverage is complete" when a less-obvious but important facet genuinely hasn't been researched yet — the report ships with a real gap that looks, from the marginal-new-finding-rate signal alone, like diminishing returns rather than incomplete coverage. **Running too long:** conversely, a genuinely broad or ambiguous research brief can keep producing marginally-new findings indefinitely (an open-ended topic has a long tail of genuinely new but decreasingly important facts), causing the heuristic to keep dispatching sub-agents well past the point of diminishing practical value, consuming budget without materially improving the report.

**Debugging:** (1) log the marginal-new-finding-rate trajectory per report and compare against final report quality (via the evaluation approach in Q11) to check whether "stopped early" reports systematically score lower on facet coverage than "ran longer" ones — if so, the threshold is too aggressive; (2) segment this analysis by research-brief type, since the right stopping point plausibly differs by domain (a narrow factual brief converges genuinely faster than an open-ended market analysis); (3) as a structural improvement, combine the marginal-finding-rate signal with an explicit facet-coverage check (did every planned sub-question receive substantive findings, not just "are new searches finding new facts") so the heuristic can distinguish "genuinely done" from "converged on the easy parts and stalled."

</details>

---

## Q20. What are the limitations of Deep Research, and how might the field evolve? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Current limitations: (1) **citation verification doesn't scale linearly with report length** (Q17) — exhaustive per-claim verification at 30-80 sources is expensive enough that most systems sample, accepting some risk of an unverified bad citation slipping through; (2) **sub-agent context isolation trades lost-in-the-middle for missed cross-cutting connections** (Q12, Q14) — there's no free lunch between the two failure modes, only a choice of which one your architecture is more exposed to; (3) **cost and latency remain high enough that most systems require explicit user opt-in** (Q15) rather than automatic routing, unlike cheaper architecture choices elsewhere in this bank; (4) **evaluation is inherently harder than single-answer RAG** (Q11) since there's no simple exact-match target, pushing most teams toward rubric-based LLM-as-judge evaluation with its own calibration challenges.

Likely evolution: tighter mid-run coordination between sub-agents (Q12's mitigation, extended into a first-class architectural feature rather than an add-on) to close the cross-cutting-connection gap without fully reverting to a single shared context; more sophisticated, facet-coverage-aware stopping criteria (Q19) replacing today's simpler marginal-new-finding heuristics; and, as RL-trained search policies (Search-R1, #42) mature, likely replacement of today's prompted sub-agent search loops (Q2) with learned search policies per sub-agent, reducing the redundant/wasted search problem (Q5's failure mode 2) at its root rather than mitigating it with coordination overhead after the fact — exactly the direction Q5's own "combining with other architectures" section anticipates.

</details>

---

## Real-World Applications

- **OpenAI Deep Research** (ChatGPT, launched Feb 2025): autonomously browses the web for roughly 5-30 minutes to produce analyst-level cited reports for finance, science, policy, and engineering research
- **Gemini Deep Research** (Google): produces comprehensive multi-source reports, now integrated with Workspace content (Gmail, Chat, Drive) and offered via an Interactions API for developers
- **LangChain `open_deep_research`**: open-source, model-agnostic deep research agent supporting configurable search tools and MCP servers
- **Hugging Face smolagents `open_deep_research`**: open replication effort benchmarked on GAIA (general AI assistant tasks), reporting ~55% pass@1 versus ~67% for OpenAI's original
- **Enterprise due-diligence and market research**: competitive landscape analysis, literature reviews, and regulatory research where a single cited answer is insufficient and a structured multi-source report is the deliverable
