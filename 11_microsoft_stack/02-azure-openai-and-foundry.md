# 02 — Azure OpenAI and Foundry for RAG

> The model layer and platform for RAG on Microsoft: deployments and quotas, embeddings, guardrails, identity and networking, evaluation, and cost control in Azure OpenAI / Microsoft Foundry.

---

> **Naming note:** Azure OpenAI Service, Azure AI Studio, Azure AI Foundry and now Microsoft Foundry are successive names around the same family of capabilities. In interviews, use the current name but show you recognise the old ones. Limits, SKUs, model names and API versions below are accurate to the Learn docs as of late 2026; always re-check the linked pages.

## Q1. What is the difference between a model and a deployment in Azure OpenAI / Foundry Models, and why does it matter for RAG code? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A **model** is the artifact (for example a GPT chat model or a `text-embedding-3` model at a specific version). A **deployment** is a named, configured instance of that model inside your resource, with a deployment type (Standard, Global Standard, Provisioned, ...), a capacity/quota allocation, and optionally a guardrail (content filter) policy attached.

In Azure OpenAI-style APIs your code calls the **deployment name**, not the model name. That has practical RAG consequences:

- You can keep a stable name like `chat-prod` and swap the underlying model version without code changes.
- Quota, content filtering and cost are per deployment, so chat, query rewriting and embeddings should usually be separate deployments.
- A typo or a deleted deployment gives a 404 `DeploymentNotFound`, a common first-day error.

OpenAI direct differs: you pass the model name and there is no deployment layer. Docs: [Deployment types](https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-models/concepts/deployment-types).

</details>

---

## Q2. Why prefer Microsoft Entra ID / managed identity over API keys when your RAG app calls Azure OpenAI? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

API keys are long-lived shared secrets: they leak into logs and repos, rotate painfully, and carry no per-caller identity. Entra ID with a **managed identity** removes the secret entirely and gives you auditable, revocable, role-based access.

- Assign the app's managed identity a role such as **Cognitive Services OpenAI User** on the resource (inference only, no management rights).
- In code, use `DefaultAzureCredential` plus a bearer-token provider for the Cognitive Services scope instead of an `api_key`.
- The same identity can be granted roles on Azure AI Search and Storage, so the whole RAG path is keyless.
- You can then disable key auth on the resource (`disableLocalAuth`) and enforce it with Azure Policy.

Trade-off: local development needs `az login` or a dev identity, and token acquisition adds a small amount of setup. Keys are still fine for a throwaway prototype, or for a gateway (such as API Management) that holds them centrally.

</details>

---

## Q3. How do you choose the embedding model and the `dimensions` setting for a RAG index on Azure OpenAI? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

You deploy an embedding model (for example `text-embedding-3-small` or `text-embedding-3-large`) as its own deployment and call the embeddings endpoint through that deployment name. The `text-embedding-3` family accepts an optional **`dimensions`** parameter that returns a shortened vector, trading a little retrieval quality for less storage, memory and latency.

| Choice | Effect |
|---|---|
| Larger model, full dimensions | Best recall; most index memory and cost |
| Smaller model | Cheaper and faster; usually fine for clean, narrow corpora |
| Reduced `dimensions` | Smaller index and faster search; measure the recall drop on your own queries |

Rules of thumb:

- The index field dimension must equal what you request; it is fixed once the index exists.
- Query-time and index-time embeddings must use the same model, version and dimensions.
- Decide by evaluating recall@k on a labelled query set, not by leaderboard scores.

Alternatives: open models such as BGE or E5 self-hosted, or Cohere/OpenAI-direct embeddings, if you need different language coverage or hosting.

</details>

---

## Q4. What is a Foundry project, and how does it relate to an Azure OpenAI resource? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A **Foundry resource** is the Azure-level account that hosts model deployments, networking, identity and guardrail policies. A **Foundry project** is a workspace inside it that groups the things one application or team needs: model connections, agents, evaluations, traces, datasets and connections to services like Azure AI Search.

Practically:

- Classic "Azure OpenAI resource" deployments are available through the Foundry resource, and the same models can be called via OpenAI-compatible endpoints.
- Projects are the unit for RBAC, evaluation runs and observability, so a RAG app typically gets one project per environment (dev, test, prod).
- Older material refers to "hubs" and "Azure AI Studio" projects; the current model is simpler, so check which generation a tutorial targets before copying code.

Learn starting point: [Microsoft Foundry documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/).

</details>

---

## Q5. How can Foundry Local and Ollama help while prototyping a RAG app before you have Azure capacity, and what are their limits? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Both let you iterate before you provision Azure capacity, by running models on your own machine.

- **Foundry Local**: a runtime that runs optimised small models (Phi-family and others) locally and exposes an OpenAI-compatible API (check the current Foundry Local docs for supported models and hardware). Good for offline work, privacy-sensitive experiments and cheap unit tests of prompt logic.
- **Ollama** (or vLLM on your own GPU): the open-source equivalent, with a larger model catalogue and the same OpenAI-style endpoint, so you can also try Phi, Llama and Mistral side by side.

Because these speak the OpenAI wire format, a thin `client factory` driven by config lets the same RAG code target a local model in dev and Azure OpenAI or a Foundry deployment in prod. Watch for differences in supported features (structured outputs, tool calling) and in answer quality: a small local model will not behave like your production model, so do final evaluation against the real deployment.

Note that hosted free tiers come and go. GitHub Models, which offered a free rate-limited OpenAI-compatible endpoint for prototyping, was retired on July 30, 2026, so do not build a workflow that depends on a free hosted endpoint staying available. Local models have no such dependency.

</details>

---

## Q6. What do Prompt Shields protect against in a RAG system, and what are the two attack types? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Prompt Shields (part of Azure AI Content Safety, surfaced as guardrails in Foundry) detect adversarial input aimed at manipulating model behaviour. They cover two attack types:

| Type | Source | RAG relevance |
|---|---|---|
| **User prompt attack** (formerly jailbreak detection) | The user's message | "Ignore your rules", role-play personas, encoded instructions |
| **Document attack** (indirect prompt injection) | Retrieved documents, emails, web pages | Hidden instructions inside content your retriever pulls into the prompt |

Per the Learn docs, user prompt attack detection is on by default while document attack detection is off by default and must be enabled in the guardrail configuration. Prompt Shields only detect injection patterns; they do not judge harmfulness, factual accuracy or groundedness, so they complement rather than replace other controls.

Docs: [Prompt Shields](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection).

</details>

---

## Q7. How do you choose between Standard, Global Standard, Data Zone and Provisioned (PTU) deployment types for a RAG workload? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Two independent choices: **where data is processed** and **how you pay for capacity**.

| Axis | Options |
|---|---|
| Processing location | Global (any Azure region), Data Zone (US, EU or APAC boundary), regional (within an Azure geography) |
| Billing | Pay-per-token (Standard) or reserved throughput (Provisioned, billed in PTUs), plus Batch at a discount for async jobs |

Guidance from the Learn docs and experience:

- **Default to Global Standard**: newest models first, broadest availability, highest default quota, lowest price.
- **Data Zone Standard** when contracts require processing to stay in the EU or US.
- **Regional Standard** when you need processing within a specific geography, accepting less quota and later model availability.
- **Provisioned** when you have steady high volume and need predictable latency; you pay for reserved PTUs whether used or not.
- **Batch** for re-indexing, offline enrichment and evaluation runs.

Note that "data at rest" stays in your resource's geography for all types; the choice governs inference processing. Not every model supports every type, so verify in [Deployment types](https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-models/concepts/deployment-types) and the models-by-region table.

</details>

---

## Q8. How do quotas (TPM/RPM) work for Azure OpenAI deployments, and how should a RAG client handle 429 responses? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Quota is allocated per subscription, region, model and deployment type in **tokens per minute (TPM)**, and you assign part of it to each deployment. A requests-per-minute (RPM) limit is derived from the TPM allocation. Exceeding either returns **HTTP 429**.

Client handling:

1. Honour the `Retry-After` header when present; otherwise use **exponential backoff with jitter**.
2. Cap retries and total wait; fail fast for interactive chat rather than queueing for a minute.
3. Reduce token pressure: smaller `max_tokens`, fewer retrieved chunks, shorter prompts. Remember that rate limiting counts the max output you reserve, not just what is generated.
4. Smooth bursts with a client-side concurrency limiter or queue.
5. Add a fallback: a second deployment in another region or a gateway that load-balances.

```python
client = AzureOpenAI(azure_endpoint=ep, azure_ad_token_provider=tp,
                     api_version=VERSION, max_retries=5)  # SDK backs off on 429
```

Separating embedding and chat deployments stops a bulk indexing job from starving user queries. See [Quotas and limits](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/quotas-limits).

</details>

---

## Q9. When would you use Azure OpenAI "On Your Data" versus building your own RAG pipeline? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

"On Your Data" let you attach a data source (typically Azure AI Search) to a chat completion call via a `data_sources` parameter, and the service handled query rewriting, retrieval and citation formatting. It was fast for demos and simple Q&A.

Status caveat: Learn documents On Your Data as **deprecated**, with Microsoft recommending Foundry Agent Service with Foundry IQ knowledge bases (or your own pipeline) and a published retirement date. As of late 2026, verify the current status on the [On Your Data page](https://learn.microsoft.com/en-us/azure/foundry-classic/openai/concepts/use-your-data) before building on it.

| Concern | Managed grounding | Own pipeline |
|---|---|---|
| Time to first demo | Hours | Days |
| Chunking, hybrid weights, reranking, filters | Limited knobs | Full control |
| Custom ACL trimming, multi-step retrieval | Hard | Natural |
| Evaluation of each stage | Opaque | Per-stage metrics |
| Lifecycle risk | Product retirement | You own maintenance |

My default for anything beyond a prototype: own the retrieval layer (Azure AI Search plus code or an orchestrator), because quality tuning and security trimming are where RAG projects succeed or fail.

</details>

---

## Q10. How do structured outputs and function/tool calling fit into a RAG pipeline? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

They make the LLM a reliable component rather than a free-text box.

- **Structured outputs** (JSON schema via `response_format`, on supported models and API versions) guarantee the response matches your schema. Use it for answers with citations (`answer`, `citations[]`, `confidence`, `insufficient_context`), for query-rewrite steps, and for LLM-judge outputs.
- **Tool/function calling** lets the model decide to call `search_docs(query, filters)` instead of you always retrieving once. This is the basis of agentic RAG: the model can reformulate, search multiple times, or skip retrieval for small talk.

Practical points:

- Validate tool arguments server-side; never let the model construct raw filter strings that bypass ACL trimming. Inject user identity filters in code.
- A schema field such as `insufficient_context: bool` gives a clean abstain path.
- Schemas have supported-keyword limits and add a first-request latency cost; check the docs for your model.
- Retrieved text is untrusted input to a model that can call tools, so tool calling raises the stakes of prompt injection (Q17).

Alternatives: Anthropic tool use and OpenAI direct offer equivalent patterns; the design is portable.

</details>

---

## Q11. Which built-in Foundry evaluators apply to RAG, and how do they map to RAGAS metrics? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Foundry's RAG evaluators split into system (final answer) and process (retrieval) evaluation:

| Foundry evaluator | Measures | Needs ground truth? | Closest RAGAS idea |
|---|---|---|---|
| Groundedness | Response consistent with context, no fabrication (precision-like) | No | Faithfulness |
| Groundedness Pro (preview) | Strict groundedness via Azure AI Content Safety, true/false | No | Faithfulness |
| Relevance | Response addresses the query | No | Answer relevancy |
| Retrieval | LLM-judged relevance of retrieved chunks | No | Context precision (judge-based) |
| Document Retrieval | NDCG, fidelity, XDCG, holes against labelled documents | Yes (relevance labels) | Classic IR metrics, context recall-style |
| Response Completeness (preview) | Covers expected information (recall-like) | Yes | Answer correctness/recall flavour |

LLM-judge evaluators take a `deployment_name` for the judge model and typically return a 1 to 5 score with a pass threshold (default 3). Concepts match RAGAS (see `01_concepts/evaluation_metrics.md`), but prompts, scales and judge models differ, so scores are not interchangeable across tools. Pick one framework per trend line. Docs: [RAG evaluators](https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/evaluation-evaluators/rag-evaluators).

</details>

---

## Q12. What levers do you have to control Azure OpenAI cost in a RAG application? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Think in tiers, from biggest wins down:

1. **Model tiering**: use a small, cheap model for query rewriting, classification, reranking prompts and routing; reserve the large model for final synthesis. Route by query difficulty.
2. **Retrieve less, better**: reranking and a smaller top-k cut input tokens, which dominate RAG cost.
3. **Prompt caching**: on supported models, repeated prompt prefixes (long system prompt, tool schemas) can be cached automatically with discounted input tokens. Put static content first and variable content (retrieved chunks, question) last. Check the docs for the minimum prefix size and supported models.
4. **Batch deployments** for re-indexing, enrichment and offline evaluation (documented at a 50% discount with a 24-hour target).
5. **Semantic or response caching** for repeated FAQs, with care for ACL and freshness.
6. **Output limits**: set `max_tokens` and ask for concise answers.
7. **Embedding dimensions** and incremental re-indexing to cut embedding and storage spend.

Measure cost per successful answer, not per request, so a cheaper model that fails more often is not mistaken for savings.

</details>

---

## Q13. How do private endpoints and data residency work together for a regulated RAG deployment on Azure OpenAI? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

They solve different problems.

- **Network isolation**: a private endpoint gives the Foundry/Azure OpenAI resource a private IP in your VNet; you then disable public network access and use private DNS so the hostname resolves privately. Do the same for Azure AI Search, Storage and Key Vault, and put the app in a VNet-integrated host. Network rules are enforced per resource.
- **Data residency**: governed by deployment type. Data at rest stays in the resource's geography, but inference may be processed in any region for Global types, within the data zone for Data Zone types, and within the geography for regional Standard/Provisioned types.

Checklist for regulated workloads:

- Pick Data Zone or regional deployments if contracts require it; do not default to Global Standard.
- Keep the vector index, logs and traces in the same boundary; observability sinks are data stores too.
- Use managed identity (Q2) and disable key auth.
- Review abuse-monitoring and data-handling terms with compliance.

Common trap: private networking is set up for the model but a trace exporter or evaluation job sends prompts to a public SaaS. See [Deployment types](https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-models/concepts/deployment-types).

</details>

---

## Q14. What should you trace and log in a Foundry-hosted RAG app, and what tooling is available? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Instrument each stage as a span so you can answer "why was this answer bad?" after the fact.

| Span | Capture |
|---|---|
| Query rewrite | Input, rewritten query, latency, tokens |
| Retrieval | Query, filters applied, chunk IDs, scores, latency |
| Rerank | Order before and after |
| Generation | Model/deployment, prompt version, tokens in/out, finish reason, guardrail annotations |
| Guardrails | Which filters fired, detected vs filtered |

Foundry projects integrate with **OpenTelemetry** and Azure Monitor / Application Insights for tracing, and traces can feed evaluators for continuous quality checks on sampled production traffic. Alternatives include Langfuse, LangSmith or Arize with the same OTel conventions.

Cautions: prompts contain user data and retrieved documents, so redact or restrict access and set retention; sample high-volume traffic; always log the deployment name and version so regressions can be tied to a model change.

</details>

---

## Q15. An EU health insurer wants a member-facing benefits assistant: EU-only processing, traffic spikes at open enrolment, and an audit trail. How do you design the model layer? `[Advanced]` `[Scenario]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Constraints: processing must stay in the EU, load is bursty, and every answer must be explainable.

**Deployment choice**
- Use **Data Zone Standard (EU)** for chat and embeddings, not Global Standard, since Global can process outside the EU. Confirm the exact models are offered in the EU data zone; availability lags Global.
- Because the spike is seasonal, avoid paying for Provisioned all year. Options: a modest regional/Data Zone Provisioned base for baseline, with spillover to Standard, or simply Standard with quota raised ahead of enrolment and a pre-load test. Decide with measured p95 latency and cost per month, not guesswork.
- Use **Data Zone Batch** for nightly re-indexing so it never competes with chat quota.

**Security and compliance**
- Private endpoints for Foundry, AI Search, Storage; public access off; managed identity; key auth disabled.
- Prompt Shields with document attack detection enabled, as benefit documents may come from partner uploads.
- Per-member ACL filters injected server-side.

**Resilience and audit**
- A gateway (for example API Management) with retry/backoff and a second EU-zone deployment as failover.
- Traces and answers with chunk IDs and prompt/model versions stored in an EU workspace with defined retention.

Trade-off to name: the strict residency choice costs newer-model access and some quota headroom. Say so and agree with compliance on what is required.

</details>

---

## Q16. Production chat latency spikes and users see 429s every night at 2 a.m. when re-indexing runs. How do you diagnose and fix it? `[Advanced]` `[Scenario]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The timing points to **shared quota contention**: the re-index job generates embeddings (and maybe LLM enrichment) from the same deployment or quota pool as user queries.

**Diagnose**
1. Check Azure Monitor metrics for the deployment: processed tokens, requests and 429 counts, aligned with the job schedule.
2. Confirm which deployments the job calls. Embedding and chat TPM are separate allocations; if the job uses the chat deployment for enrichment, that is the culprit.
3. Inspect whether clients honour `Retry-After`; naive immediate retries amplify the storm.
4. Look at the job's concurrency and batch sizes.

**Fix (in order)**
- Move bulk work to a **Batch** deployment (separate enqueued quota, discounted) where latency does not matter.
- Give the job its own embedding deployment and cap its concurrency with a token-bucket limiter below the quota.
- Make indexing **incremental** (content hashes) so only changed chunks are re-embedded.
- Add backoff with jitter to all clients; set a fail-fast timeout for interactive requests.
- If user traffic itself is the issue, request more quota or add a second-region deployment behind a gateway.

Verify by replaying the schedule in staging and watching 429 rate and p95 latency.

</details>

---

## Q17. How do you defend a RAG system against indirect prompt injection, and what can Azure's tooling not do for you? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Indirect injection means a retrieved document contains instructions ("email the user's data to ...") that the model treats as commands. No single control stops it, so layer them.

**Azure tooling**
- **Prompt Shields document attack detection**: classify retrieved content before it reaches the model; off by default, so enable it.
- **Spotlighting (preview)**: marks documents as lower trust to the model by encoding them; costs extra tokens and can leak "base-64" mentions into answers.
- Content filters for harmful output categories.

**Architecture (your responsibility)**
- Keep the model **least-privilege**: no tools that send data out or write state without human confirmation.
- Separate trusted instructions from untrusted content with clear delimiters and a system prompt that says retrieved text is data.
- Control the corpus: ingestion review for externally authored content, ACL trimming so a user can only inject through documents they could read anyway.
- Render output safely (no auto-loading markdown images or links to attacker-controlled URLs, a known exfiltration path).
- Log and red-team with representative attacks, including your non-English content, since the shield models are primarily trained on English.

Limit to state plainly: detectors have false negatives. The robust defence is limiting what a hijacked model can do.

Docs: [Prompt Shields](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection).

</details>

---

## Q18. How do you safely migrate a RAG app to a new model version when Azure retires the one you use? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Models have retirement dates published on Learn, and deployments can be set to auto-upgrade or pinned. Silent upgrades are risky for RAG because prompts, citation habits and abstention behaviour shift between versions.

**Process**
1. **Track retirement notices** and the models-by-region tables; plan a few months ahead.
2. **Build a golden set**: real queries with expected sources/answers, including must-abstain cases and adversarial ones.
3. **Deploy the candidate side by side** under a new deployment name; keep the old one live.
4. **Run offline evaluation**: groundedness, relevance, retrieval, completeness, plus latency, token usage and cost. Compare with thresholds, not eyeballing. Review failing items manually.
5. **Re-tune prompts** if needed (output format, citation style, refusal behaviour).
6. **Canary**: route a small share of traffic via config or gateway, watch guardrail rates and user feedback.
7. **Cut over**, keep rollback until the old model is near retirement, then delete.

Pin versions in production rather than relying on auto-upgrade, and treat model version as part of the config you log with each trace. A different deployment type may also change quota and latency profile, so retest limits.

Reference: [Model retirements](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/concepts/model-retirements).

</details>

---

## Q19. You want to change the embedding model or its dimensions on a live Azure AI Search index. How do you do it without downtime or silent quality loss? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Vectors from different models, versions or dimension settings are **not comparable**, and a vector field's dimension count is fixed at creation. So this is a re-index, not a config flip.

**Blue/green approach**
1. Create a **new index** (or new vector field) with the new dimension and algorithm settings.
2. Re-embed the corpus into it using a dedicated embedding deployment and, where latency is irrelevant, batch or throttled jobs (see Q16).
3. Run the golden query set against both indexes. Compare recall@k and NDCG with the Foundry Document Retrieval evaluator or your own labels, then downstream groundedness.
4. Switch the query side atomically: query embedding deployment and index alias must change together. A mismatch yields plausible but wrong results with no error.
5. Keep the old index for rollback, then delete it.

**Pitfalls**
- Query embeddings must use identical model, version and `dimensions` as the index.
- Hybrid weights and score thresholds tuned on the old model often need retuning.
- Budget: re-embedding the whole corpus costs tokens and time.

Mitigation: store the embedding model and dimension in index metadata and in every trace, so mismatches are detectable.

</details>

---

## Q20. What is the difference between groundedness detection in Azure AI Content Safety and the Foundry groundedness evaluator, and where does each belong? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

They answer a similar question, "is the answer supported by the sources?", at different points in the lifecycle.

| | Content Safety groundedness detection | Foundry Groundedness / Groundedness Pro evaluator |
|---|---|---|
| Purpose | Runtime guardrail | Offline or sampled quality measurement |
| Inputs | Generated text plus grounding sources (and optionally query) | `response`, `context`, optionally `query` |
| Output | Ungrounded flag and spans for a single response | Pass/fail with a score or label per dataset row |
| Cost / latency | Adds a call per response in the request path | Runs outside the user path |
| Action | Block, regenerate, add a warning | Gate releases, track trends |

Design guidance:

- Use the **evaluator** on golden sets in CI and on sampled production traces; it informs prompt, retrieval and model decisions.
- Use **runtime detection** only where a wrong answer is costly (medical, legal, financial) and the added latency is acceptable. Pair it with a fallback such as "I could not verify this in the documents".
- Neither verifies that the source itself is correct, and both can err; periodically calibrate them against human labels.

Groundedness Pro in the evaluator suite uses the Content Safety service. Feature availability (preview status, supported regions and domains) changes, so confirm in current docs.

</details>

---

## Q21. Field technicians at a utility often have no connectivity and need answers from repair manuals. How would you architect RAG with Foundry Local or Phi-class models? `[Advanced]` `[Scenario]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Constraint: offline-first, modest hardware, but answers must be trustworthy for safety-relevant tasks.

**On-device stack**
- A small language model (Phi-family or similar) served locally, for example through Foundry Local, which exposes an OpenAI-compatible API. Alternatives: Ollama or llama.cpp-based runtimes.
- A small local embedding model and an embedded vector store (SQLite-based, FAISS, or similar) holding a **pre-built, versioned manual index** synced when online.
- Same app code as the cloud path via an OpenAI-compatible client and config switch.

**Design choices**
- Keep answers **extractive and cited**: show the manual section alongside the text, because small models hallucinate more. Use a strict "answer only from context, otherwise say not found" prompt and structured output if the runtime supports it.
- Limit corpus per device (model, region) to keep retrieval precise on weak hardware.
- **Hybrid routing**: when connected, send hard or ambiguous queries to a cloud deployment; offline, fall back to local and label the answer as such.
- Sync index and model updates with version stamps; log queries locally and upload later.

**Trade-offs**: lower reasoning quality, device storage and battery, harder update management. Evaluate the local stack on the same golden set as the cloud one, and only ship if groundedness meets a threshold you define for safety-critical content.

</details>

---

## Q22. Under what conditions do Provisioned throughput (PTU) deployments beat pay-as-you-go for RAG, and how do you size and de-risk them? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

PTU deployments reserve model processing capacity, billed per PTU-hour whether or not you use it. They win when **utilisation is high and steady** and when **latency consistency** matters; they lose for spiky or low-volume traffic.

**Break-even thinking**
- Estimate sustained tokens per minute (input and output separately, since output weighs more heavily in PTU accounting) and compare the PTU cost for that throughput with pay-as-you-go cost for the same tokens. Use the capacity calculator and per-model PTU guidance in [Provisioned throughput concepts](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/concepts/provisioned-throughput); minimum PTUs and throughput per PTU differ by model.
- If average utilisation of the reservation is low, Standard is cheaper. If it is consistently high, PTU usually is.

**Sizing and de-risking**
1. Load-test with a production-shaped prompt mix (RAG prompts are input-heavy, and cache hits change the picture).
2. Size PTU for baseline, not peak; handle peaks by **spillover** to a Standard deployment or client-side routing when PTU returns 429.
3. Prefer Global or Data Zone Provisioned where residency allows, since capacity is easier to obtain than regional; check availability before committing.
4. Consider reservations only after a few months of stable usage data.
5. Monitor provisioned utilisation and re-size when prompt shapes change, for example after raising top-k.

Non-Azure comparison: self-hosting on vLLM gives similar fixed-cost economics with more ops burden.

</details>

---

## Q23. A RAG service gets HTTP 429 from an Azure OpenAI deployment and the response includes a `Retry-After` header. What should the client do first? `[Intermediate]` `[MCQ]`

- A. Immediately resend the same request on a new connection to bypass the limiter
- B. Wait at least the indicated time, then retry with jittered backoff and a retry cap
- C. Switch from Entra ID to API key authentication to get a larger limit
- D. Double `max_tokens` so each retry returns more content per call

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Correct: B.** A 429 means the deployment's tokens-per-minute or requests-per-minute allocation was exceeded, and the header tells you when capacity is likely available. Waiting, adding jitter so many clients do not retry in lockstep, and capping retries keeps the system stable and the user experience bounded. Option A makes the overload worse, since rate limits are tracked per deployment, not per connection. Option C is wrong because the authentication method does not change quota; limits are tied to the deployment and subscription allocation. Option D increases tokens reserved per request, which consumes more of the limit and makes further 429s more likely.

</details>

---

## Q24. You request `dimensions=256` from a `text-embedding-3` deployment for a new index. What must hold for retrieval to behave correctly? `[Intermediate]` `[MCQ]`

- A. Only documents need 256 dimensions, since queries are re-projected automatically at search time
- B. The index vector field can stay at 1536 dimensions, and the search service pads the shorter vectors
- C. Any embedding model can embed queries, as long as vectors are normalised before search
- D. The index field and both document and query embeddings must use the same model and 256 dimensions

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Correct: D.** Similarity scores are only meaningful between vectors from the same embedding space with the same dimensionality, and a vector field's dimension count is fixed in the index schema. Option A is wrong because nothing re-projects queries; a query embedded at a different size is rejected or compares incorrectly. Option B is wrong because search services validate vector length against the field definition rather than padding, and padding would distort similarity anyway. Option C is wrong because normalisation does not align different models' spaces; mixing models gives plausible-looking but meaningless rankings.

</details>

---

## Q25. Your RAG app retrieves emails from external senders. Which Azure guardrail setting specifically targets hidden instructions inside those retrieved documents? `[Intermediate]` `[MCQ]`

- A. Prompt Shields document attack detection, which must be enabled in the guardrail configuration
- B. The hate and violence severity filter, set to its strictest level
- C. A larger `top_k` so malicious chunks are diluted by benign ones
- D. Lowering `temperature` to zero so the model cannot follow embedded commands

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Correct: A.** Document attack detection scans content submitted as documents for injected instructions, which is exactly the indirect injection pattern, and the Learn docs note it is off by default, so it must be turned on. Option B is wrong because harm-category filters judge content like hate or violence, and an injected instruction can be polite and harmless-looking. Option C is wrong because dilution does not neutralise an instruction; one malicious chunk in context can still steer the model. Option D is wrong because temperature affects sampling randomness, not whether a model obeys instructions; a deterministic model follows injected commands just as faithfully.

</details>

---

## Q26. A bank requires that Azure OpenAI prompts and responses are processed only inside the EU, but wants more quota than a single-geography Standard deployment offers. Which deployment type fits best? `[Advanced]` `[MCQ]`

- A. Global Standard, because it has the highest default quota
- B. Global Batch, because it has a separate token quota
- C. Data Zone Standard configured for the EU data zone
- D. Developer tier, because it is the cheapest

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Correct: C.** Data Zone deployments keep inference processing within the chosen zone (the EU, following the Azure EU Data Boundary) while typically offering higher default quota than geography-bound Standard, because traffic can be routed across datacenters in the zone. Option A has the largest quota but Global types may process data in any Azure region, violating the requirement. Option B is also global and asynchronous with a 24-hour target, so it is unsuitable for interactive chat even though its quota is separate. Option D is intended for evaluating fine-tuned models only, has a 24-hour lifetime, no SLA and no data residency guarantee. Verify model availability per data zone in the current docs.

</details>

---
