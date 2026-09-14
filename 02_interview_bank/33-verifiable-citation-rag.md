# 33 — Verifiable / Citation RAG

> Every claim in the generated answer is linked to a specific retrieved passage — and that link is verified, not assumed.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
Query
    │
    ▼
Retriever (fetches candidate source chunks)
    │
    ▼
Citation-aware Generator (produces answer with inline citation markers per claim)
    │
    ▼
Attribution / Entailment Verifier (checks each cited claim against its source chunk via NLI or LLM-judge)
    │
    ▼
Citation Renderer (formats verified citations; flags or removes unsupported claims)
```

### Key Components

| Component | Responsibility |
|---|---|
| Retriever | Fetches candidate source chunks relevant to the query |
| Citation-aware Generator | Produces the answer with inline citation markers tied to specific passages |
| Attribution/Entailment Verifier | Checks whether the cited passage actually entails the paired claim |
| Citation Renderer | Formats verified citations in the final output, or flags/removes unsupported claims |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Retriever + Generator stack | Standard dense retriever paired with an LLM generator |
| NLI verification model | `cross-encoder/nli-deberta-v3-base`, `bart-large-mnli` |
| LLM-as-judge verification | Cheap model (e.g. Claude Haiku) prompted for SUPPORTED / NOT_SUPPORTED verdicts |
| Evaluation benchmark | ALCE (attribution scoring benchmark, Gao et al., 2023) |
| Post-processing | Citation-formatting and unsupported-claim flagging post-processor |

---

## Q1. What is Verifiable RAG and why is "citing a source" not the same as "grounding a claim"? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Verifiable RAG** (also called Citation RAG or Attributed RAG) extends standard RAG with two additional requirements:
1. Every factual claim in the output is paired with a citation pointing to a specific retrieved passage
2. That citation is **verified** — the passage actually supports the claim

**The grounding gap — why standard RAG citations often fail:**

In standard RAG, asking the LLM to "cite your sources" often produces hallucinated or inaccurate citations:

```
Hallucinated citation:
  Claim:    "RAG reduces hallucinations by 37%."
  Citation: [Source 2]
  Reality:  Source 2 discusses RAG architecture — never mentions 37%
```

```
Correct citation:
  Claim:    "RAG reduces hallucinations by 37%."
  Citation: [Source 2, paragraph 3]
  Verification: Source 2 paragraph 3 says "...reduced hallucination rate by 37%..."
  → Supported ✓
```

**Three levels of citation quality:**

| Level | Description | Failure Mode |
|-------|-------------|--------------|
| **Source-level** | Answer cites a document | LLM may cite document that doesn't support the claim |
| **Passage-level** | Answer cites a specific chunk | LLM may misattribute which sentence supports the claim |
| **Span-level** | Answer cites the exact span | Highest precision; requires span extraction |

**When Verifiable RAG is required:**

- Medical and legal contexts where incorrect citations create liability
- Research assistants where users follow citations to verify claims
- Enterprise compliance reporting where traceability is audited
- Financial analysis where specific figures must trace to specific documents

</details>

---

## Q2. How do you generate passage-level citations in a RAG response? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Method 1: In-context citation instructions**

Instruct the LLM to place inline citation markers after each claim, then map them back to retrieved passages.

```python
from anthropic import Anthropic

client = Anthropic()

def generate_with_citations(query: str, passages: list[dict]) -> dict:
    """
    passages: list of {"id": int, "text": str, "source": str}
    Returns: {"answer": str, "citations": list[{"claim": str, "passage_id": int}]}
    """
    passages_block = "\n\n".join(
        f"[{p['id']}] {p['text']}"
        for p in passages
    )
    
    prompt = f"""You are a precise research assistant. Answer the question using
the numbered passages below. After each factual claim in your answer, insert
a citation in brackets like [1] or [2] referencing the passage number.
Only cite passages that directly support the specific claim.
Do not add a citation if no passage supports the claim — instead, omit the claim.

Passages:
{passages_block}

Question: {query}

Answer (with inline citations):"""

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# Example output:
# "RAG was introduced in 2020 [1] and has been shown to reduce hallucinations
#  significantly [3]. The retrieval component typically uses a bi-encoder
#  architecture [2]."
```

**Method 2: Structured citation output**

Force structured output with explicit claim → citation mappings:

```python
import json

STRUCTURED_PROMPT = """Answer the question. For each factual claim, output JSON:
{{
  "claims": [
    {{"claim": "exact sentence from your answer", "passage_ids": [1, 3]}}
  ],
  "answer": "full answer text with [1],[2] inline markers"
}}

Passages:
{passages}

Question: {query}"""

def generate_structured_citations(query: str, passages: list[dict]) -> dict:
    resp = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=2048,
        messages=[{"role": "user", "content": STRUCTURED_PROMPT.format(
            passages="\n".join(f"[{p['id']}] {p['text']}" for p in passages),
            query=query
        )}]
    )
    return json.loads(resp.content[0].text)
```

</details>

---

## Q3. How do you verify that a citation actually supports a claim? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Generating citations is easy; verifying them requires a separate **attribution verification** step.

**Method 1: NLI-based verification (Natural Language Inference)**

Use an NLI model to check whether the cited passage *entails* the claim.

```python
from transformers import pipeline

nli = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-base")

def verify_citation(claim: str, cited_passage: str) -> dict:
    """Returns entailment/neutral/contradiction + confidence."""
    result = nli(f"{cited_passage} [SEP] {claim}")[0]
    return {
        "claim": claim,
        "passage": cited_passage,
        "label": result["label"],      # ENTAILMENT / NEUTRAL / CONTRADICTION
        "confidence": result["score"],
        "supported": result["label"] == "ENTAILMENT" and result["score"] > 0.7
    }

# Example:
claim = "RAG reduces hallucinations by 37%."
passage = "In our experiments, RAG-equipped models showed a 37% reduction in hallucination rate."
result = verify_citation(claim, passage)
# → {"label": "ENTAILMENT", "confidence": 0.94, "supported": True}
```

**Method 2: LLM-as-judge verification**

```python
VERIFY_PROMPT = """Does the following passage support the claim?
Answer with 'SUPPORTED', 'NOT_SUPPORTED', or 'PARTIALLY_SUPPORTED'.

Claim: {claim}

Passage: {passage}

Verdict:"""

def llm_verify_citation(claim: str, passage: str) -> str:
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",   # cheap model for binary verification
        max_tokens=10,
        messages=[{"role": "user", "content": VERIFY_PROMPT.format(
            claim=claim, passage=passage
        )}]
    )
    return resp.content[0].text.strip()
```

**Full verification pipeline:**

```python
def verifiable_rag(query: str, passages: list[dict]) -> dict:
    # Step 1: Generate answer with citations
    raw = generate_structured_citations(query, passages)
    
    # Step 2: Verify each citation
    verified_claims = []
    for item in raw["claims"]:
        claim = item["claim"]
        evidence = []
        for pid in item["passage_ids"]:
            passage_text = next(p["text"] for p in passages if p["id"] == pid)
            verdict = verify_citation(claim, passage_text)
            evidence.append(verdict)
        
        # Claim is supported if at least one cited passage entails it
        supported = any(e["supported"] for e in evidence)
        verified_claims.append({
            "claim": claim,
            "supported": supported,
            "evidence": evidence,
        })
    
    return {
        "answer": raw["answer"],
        "verified_claims": verified_claims,
        "unsupported_claims": [c for c in verified_claims if not c["supported"]],
    }
```

**What to do with unsupported claims:**

1. **Remove them:** Regenerate the answer without the unsupported claims
2. **Flag them:** Show the answer with a warning on flagged claims
3. **Retrieve more:** Trigger additional retrieval to find supporting evidence
4. **Abstain:** If the key claim cannot be verified, don't answer

</details>

---

## Q4. How do you evaluate a Citation RAG system? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Citation RAG requires metrics beyond standard RAG evaluation because it has an additional attribution quality dimension.

**Metric 1: Citation Precision** — what fraction of generated citations actually support their paired claim

```python
def citation_precision(verified_claims: list[dict]) -> float:
    if not verified_claims:
        return 0.0
    supported = sum(1 for c in verified_claims if c["supported"])
    return supported / len(verified_claims)
```

**Metric 2: Citation Recall** — what fraction of verifiable claims in the answer have at least one citation

```python
def citation_recall(claims: list[dict]) -> float:
    """claims: list with 'has_citation' and 'is_factual' flags."""
    factual_claims = [c for c in claims if c["is_factual"]]
    if not factual_claims:
        return 1.0
    cited = sum(1 for c in factual_claims if c.get("has_citation", False))
    return cited / len(factual_claims)
```

**Metric 3: Attribution F1** — harmonic mean of citation precision and recall

**Metric 4: Claim Faithfulness** — across all supported claims, does the answer accurately reflect what the passage says (no distortion)?

```python
FAITHFULNESS_PROMPT = """On a scale of 1-5, how faithfully does the claim
represent the meaning of the passage? 5 = exact paraphrase, 1 = distortion.

Claim: {claim}
Passage: {passage}
Score (1-5):"""
```

**ALCE benchmark** (Gao et al., 2023) provides:
- Automatic citation evaluation using NLI-based attribution scoring
- Human-annotated citation quality labels for calibration
- Three sub-tasks: ASQA (open-domain), QAMPARI (multi-answer), ELI5 (explanations)

**Production monitoring:**

```python
# Track citation quality per query in production
metrics = {
    "citation_precision": compute_citation_precision(response),
    "unsupported_claim_rate": len(response["unsupported_claims"]) / len(response["verified_claims"]),
    "citation_coverage": len(response["claims_with_citations"]) / len(response["all_claims"]),
}
```

</details>

---

## Q5. What is the difference between attribution and hallucination detection in RAG? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

These are related but distinct problems:

```
Hallucination detection: "Did the LLM invent a fact not present in any source?"
Attribution verification: "Does this specific citation support this specific claim?"

Relationship:
  ─ A claim can be hallucinated AND incorrectly cited  (worst case)
  ─ A claim can be real but incorrectly cited          (citation error, not hallucination)
  ─ A claim can be correctly cited but misrepresented  (faithfulness error)
  ─ A claim can be correct and correctly cited         (ideal case)
```

**Hallucination detection approach:**

Check whether any retrieved passage supports the claim — regardless of which passage the LLM cited.

```python
def detect_hallucination(claim: str, all_passages: list[str]) -> bool:
    """A claim is hallucinated if NO passage in the retrieved set supports it."""
    for passage in all_passages:
        result = verify_citation(claim, passage)
        if result["supported"]:
            return False   # at least one passage supports it → not hallucinated
    return True   # no passage supports it → hallucinated
```

**Attribution verification approach:**

Check whether the *specific cited* passage supports the claim.

```python
def check_attribution(claim: str, cited_passage: str) -> bool:
    """Attribution fails if the cited passage doesn't support the claim,
    even if another passage would."""
    return verify_citation(claim, cited_passage)["supported"]
```

**Why both matter:**
- A hallucination detector that passes a correctly-attributed claim but misses a hallucinated uncited claim will underreport hallucinations
- An attribution checker that only checks cited passages won't catch hallucinated claims that happen to have no citation at all

In production Verifiable RAG, run both: (1) verify all citations, (2) check all uncited factual claims against the full retrieved set.

</details>

---

## Q6. Walk through the Verifiable RAG architecture end-to-end. `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

```
Query
    │
    ▼
Retriever (fetches candidate source chunks)
    │
    ▼
Citation-aware Generator (produces answer with inline citation markers per claim)
    │
    ▼
Attribution / Entailment Verifier (checks each cited claim against its source chunk)
    │
    ▼
Citation Renderer (formats verified citations; flags or removes unsupported claims)
```

The critical architectural addition over standard RAG is the verifier stage: a citation-aware generator alone (Q2) only produces *claimed* attributions — it has no independent check that the model actually followed its own citation instructions correctly. The verifier re-examines each (claim, cited passage) pair using a method separate from the generator itself (NLI or a second LLM call, Q3), so a citation's correctness is confirmed by a distinct process rather than trusted because the generator was asked nicely to be accurate. The renderer then acts on that verdict — flagging, removing, or triggering re-retrieval for anything that didn't pass (Q3's four response options).

</details>

---

## Q7. What is the single distinctive mechanism that separates Verifiable RAG from standard RAG's "cite your sources" prompting? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The distinctive mechanism is an **independent verification step that checks the generator's own citation claims**, rather than trusting the generator to have cited correctly just because it was instructed to. Standard RAG's "cite your sources" prompting produces citations, but a citation is just another piece of generated text — the model can hallucinate a citation exactly as readily as it can hallucinate a fact, and nothing in a bare prompting approach catches this (Q1's hallucinated-citation example).

Verifiable RAG closes this gap with a second, independent process — NLI entailment checking or an LLM-as-judge call (Q3) — that re-examines whether the cited passage actually supports the claim, after generation has already happened. This is the same "generate, then independently verify" pattern used elsewhere in this bank (SURGE's NLI grounding, #40) applied specifically to free-form prose citations rather than structured schema fields (Q8) — the citation is only trusted once it has passed a check the generator itself had no hand in producing.

</details>

---

## Q8. How does Verifiable RAG compare to SURGE's (#40) per-field grounding? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Both architectures use the same core verification primitive — an NLI or LLM-judge check of whether a cited/source passage entails a claimed value — but apply it to different output shapes. Verifiable RAG verifies **claims embedded in free-form prose**, where the harder sub-problem is often just *identifying* what the discrete claims even are before they can be checked (Q2's claim-extraction step). SURGE verifies **individual fields in a predefined schema**, where the claim boundaries are already fixed by the schema itself (Q40's `field_sources`), making claim identification trivial but requiring upfront schema design.

Choose Verifiable RAG for open-ended question answering and research assistants where the answer shape isn't known in advance; choose SURGE when the output needs to populate a fixed, machine-readable structure (a database row, an API response). A system needing both — a structured extraction that also includes a free-form summary — can combine per-field NLI grounding (SURGE) for the structured portion with claim-level attribution verification (this file) for the prose portion.

</details>

---

## Q9. What is the research origin of citation/attribution verification in RAG? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

ALCE (Gao et al., 2023, *Enabling Large Language Models to Generate Text with Citations*) is the benchmark most directly associated with formalizing attribution evaluation for long-form RAG answers, introducing automatic NLI-based attribution scoring alongside human-annotated citation quality labels across three sub-tasks: ASQA (open-domain question answering), QAMPARI (multi-answer questions), and ELI5 (long-form explanations). The benchmark's contribution was establishing that "does the model cite something" and "is the citation actually correct" are measurably different quantities, and that models scoring well on standard answer-quality metrics can still score poorly on attribution — motivating architectures that treat verification as a first-class pipeline stage rather than an assumed byproduct of good citation prompting.

This connects to the broader natural language inference (NLI) research this file's verification methods draw on (Q3) — entailment classification (SNLI, MNLI datasets) predates RAG entirely, and ALCE's contribution was specifically applying that existing NLI machinery to the citation-verification problem at scale, in the same way SURGE (#40, Q19) later applied it to structured field-level grounding.

</details>

---

## Q10. What are the key tuning knobs for citation verification, and how do you choose them? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Knob | Effect | Starting point |
|---|---|---|
| NLI ENTAILMENT confidence threshold (Q3's `> 0.7`) | Higher = stricter, fewer false-positive "supported" verdicts | 0.7 is a reasonable default; raise for compliance-sensitive domains (Q18) |
| Citation granularity (Q1's source/passage/span levels) | Coarser granularity is cheaper to generate and verify but has a higher misattribution risk | Passage-level is the practical default; span-level for domains where exact traceability matters most |
| Verification method (NLI vs. LLM-as-judge, Q11) | NLI is cheaper and more consistent; LLM-judge handles more nuanced/implicit entailment | Start with NLI for cost; escalate to LLM-judge for claims NLI scores as NEUTRAL/uncertain |
| Multi-passage attribution handling (Q13) | Whether a claim can be marked supported by combining evidence across passages, or only by a single passage | Single-passage-only is simpler and stricter; multi-passage support requires more verification logic but better matches how complex claims are often actually grounded |

The confidence threshold and granularity interact: span-level citations are inherently easier to verify with high confidence (the exact text either appears or doesn't), while source-level citations require the verifier to essentially search the entire document for support, which is both more expensive and produces less confident verdicts — finer granularity and a higher confidence bar tend to go together in well-tuned systems.

</details>

---

## Q11. How do you decide between NLI-based and LLM-as-judge verification in production? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

NLI-based verification (`cross-encoder/nli-deberta-v3-base`, Q3) is cheap (milliseconds per pair, runs on CPU) and produces consistent, well-calibrated verdicts for claims where entailment is relatively direct — but it can struggle with claims requiring world knowledge or multi-step inference to connect to the passage (a claim that's *implied* by the passage rather than directly stated). LLM-as-judge verification (Q3's `llm_verify_citation`) is more expensive and less consistent run-to-run, but handles nuanced or implicit entailment relationships that a smaller NLI model misses, and can be prompted to explain its reasoning for audit purposes.

The practical production pattern is tiered verification: run the cheap NLI check first on every citation; escalate only the NEUTRAL or low-confidence cases to the more expensive LLM-judge check, rather than running the expensive check on every claim. This mirrors the general pattern of using a cheap model for high-volume, low-ambiguity work and a stronger model only where the cheap method's confidence is insufficient — the same tiering principle applied elsewhere in this bank (e.g., ToT-RAG's Haiku-for-generation, Sonnet-for-synthesis split, #37).

</details>

---

## Q12. What is the characteristic failure mode when citation granularity is too coarse? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Source-level citations (Q1's coarsest level) only claim "this document supports this statement" without pointing to which part of a potentially long document actually does — this passes an easy verification bar (does *any* part of the document entail the claim?) that can mask a genuinely wrong attribution: a document containing 20 paragraphs might have exactly one sentence relevant to the claim, and a coarse verifier checking "does this document broadly relate to this claim" can return a false "supported" verdict even when the specific reasoning connecting document to claim doesn't actually hold.

**Symptom:** citation precision (Q4's metric) looks artificially high at source-level granularity compared to what passage- or span-level verification of the same answers would reveal, because coarse verification is systematically more permissive — this is a case where a metric can be technically well-defined and still misleading if the granularity it operates at doesn't match what users actually need (the ability to click a citation and immediately see the supporting text, not skim an entire document). **Mitigation:** tighten citation granularity to passage- or span-level wherever verification cost allows (Q10), and if source-level citation is unavoidable for cost reasons, report citation precision at whatever the *coarsest* level actually used is, rather than implying a stricter guarantee than the system provides.

</details>

---

## Q13. How do you handle a claim that's supported by combining multiple passages rather than a single one? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Some claims are genuinely only supported by synthesizing information across several retrieved passages — "Company X's revenue grew 18% while its main competitor's fell 5%" might require one passage for each company's figures, with neither passage alone entailing the combined comparative claim. A verifier checking each cited passage independently (Q3's default pattern) will correctly find that *neither* passage alone entails the full compound claim, potentially flagging a genuinely well-grounded claim as unsupported.

**Handling:** (1) decompose compound claims into their atomic sub-claims before verification — "Company X grew 18%" and "Competitor fell 5%" each verify independently against their respective single passage, and the compound comparative statement is then treated as correctly grounded if both atomic sub-claims pass; (2) for genuinely irreducible multi-passage claims (a claim that only makes sense as a synthesis, not a decomposable conjunction), concatenate the cited passages and verify the claim against the *combined* text rather than each passage individually, since NLI models can accept multi-sentence premises. The generator's citation step (Q2) needs to support multi-passage citations (`passage_ids: [1, 3]` in the structured output) for this to work at all — a citation format restricted to one passage per claim structurally cannot represent legitimately multi-source claims.

</details>

---

## Q14. How do you scale citation verification to long-form, many-claim answers without the verification cost exploding? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A long-form answer (a multi-paragraph research summary, a Deep Research report, #43) can contain dozens of individual claims, each needing its own verification call — at NLI's per-pair cost this is cheap in aggregate, but at LLM-judge cost (Q11) for every claim, verification can become a meaningful fraction of total pipeline cost and latency.

The same tiering principle from Q11 is the primary lever, extended with claim-importance weighting: verify every claim with the cheap NLI pass, but reserve the expensive LLM-judge escalation specifically for claims flagged as **load-bearing** — numeric figures, direct quotes, or claims a downstream reader is most likely to rely on or challenge — rather than escalating every NLI-uncertain claim uniformly (this is the same load-bearing-claim sampling strategy Deep Research RAG's Q17 uses for report-scale citation checking, applied here at the single-answer scale). Batch NLI verification calls together (multiple claim-passage pairs in one batched inference call) rather than looping one at a time, since cross-encoder throughput benefits substantially from batching, exactly as noted for SURGE's per-field validation (#40, Q15).

</details>

---

## Q15. What happens if a genuinely correct claim has no retrievable supporting passage at all? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Verification can only ever check citations against what was actually retrieved — if the retriever missed a genuinely relevant passage (a standard retrieval-recall failure, unrelated to the verification logic itself), a true claim has no passage to cite, and the generator faces a choice: state the claim uncited (failing citation recall, Q4's second metric), fabricate a citation to the nearest-but-not-quite-right passage (a hallucinated citation that verification should catch and reject), or omit the claim entirely (the safest default per Q3's prompt instruction to "omit the claim" rather than cite unsupported material).

This is fundamentally a **retrieval problem wearing a citation-verification costume** — no amount of stricter verification logic fixes a claim that's true but unretrievable; the fix is upstream, in retrieval quality (better query formulation, hybrid search, larger `k`) rather than downstream in the verifier. The practical signal to watch for: a high rate of correctly-omitted claims (tracked via a human/judge review comparing the answer against what a comprehensive answer *should* have included) points at a retrieval recall gap, not a verification gap — a distinction worth making explicitly in production monitoring, since the two point to completely different fixes (improve retrieval vs. improve the verifier).

</details>

---

## Q16. How would you build a decision-gate benchmark to certify a Verifiable RAG deployment's citation quality before shipping to a compliance-sensitive domain? `[Advanced]` `[Scenario]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

For domains where an incorrect citation creates liability (Q1: medical, legal, financial), "citation precision looks good on average" isn't sufficient — the deployment needs an explicit, auditable threshold:

```
1. Build a held-out labeled set: (query, answer, claim, cited passage, human
   verdict) tuples, ideally including deliberately adversarial cases (a
   passage that's topically related but doesn't actually support the
   specific claim -- the exact case NLI verification is meant to catch).

2. Define acceptance criteria with compliance sign-off: minimum citation
   precision (e.g., >=98% for a legal-brief use case, given the cost of
   even one confidently-wrong citation), minimum citation recall (every
   material factual claim must be cited, not just easy ones), and a
   maximum unsupported-claim rate that reaches the end user unflagged.

3. Run the full pipeline against the held-out set; compute citation
   precision/recall/F1 (Q4) and compare against thresholds.

4. Gate: FAIL if precision or recall falls below threshold, or if the
   verifier's own false-positive rate (Q19) on the adversarial cases
   exceeds an acceptable bound. Record measured numbers in the deployment
   manifest so a regression is visible on the next evaluation run's diff.

5. Re-run this gate on every change to the generator prompt, the
   verification model/threshold, or the retriever -- any of the three
   can shift citation quality independently.
```

The key discipline, as in SURGE's equivalent gate (#40, Q12), is per-domain and per-claim-type thresholds rather than one global number — a legal-citation use case's acceptable precision bar is meaningfully different from a casual research-assistant use case's, and conflating them into a single target either over-constrains the low-stakes case or under-protects the high-stakes one.

</details>

---

## Q17. What security risks are specific to citation verification? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

- **Adversarial passages crafted to pass NLI entailment for false claims** — since the verifier only checks whether the *cited passage* entails the *claim*, an attacker who can influence corpus content could plant a passage explicitly worded to entail a false statement (e.g., a document stating "Company X's revenue grew 200% in Q3" when the true figure is far lower) — the verifier correctly confirms the citation is accurate to its source, without any way to know the source itself is fabricated. This is the same source-truthfulness-vs-internal-consistency gap noted for SURGE (#40, Q17): verification confirms the claim matches the citation, never that the citation is itself trustworthy.
- **NLI model blind spots exploited via phrasing** — NLI models can be less reliable on negation, numerical precision, or entailment involving units and quantities (a passage saying "under $5B" entailing or not entailing a claim of "$4.2B" is a genuinely subtle case), and an adversarial or simply ambiguous passage exploiting a known NLI weakness could produce an incorrect verdict in either direction.
- **Verification-bypass via claim rephrasing** — a generator (whether through genuine model behavior or adversarial prompting) that rephrases a claim just enough to shift it from NEUTRAL to a borderline ENTAILMENT verdict, without the underlying factual content actually being any better supported, can game a threshold-based verifier without the claim's actual grounding improving at all.

Mitigation follows the same defense-in-depth pattern used throughout this bank: pair citation verification with source-trust scoring on the underlying corpus (verification alone never substitutes for corpus-level trust), periodically audit the verifier itself against known-hard NLI cases (negation, numerical precision) to catch systematic blind spots, and treat any threshold-based verifier as gameable in principle, monitoring for claims that cluster suspiciously close to the acceptance threshold as a possible gaming signal.

</details>

---

## Q18. Design a Verifiable RAG system for a legal research assistant. `[Advanced]` `[Scenario]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Requirements:** every statement in a generated legal brief must be attributable to a specific cited case or statute; incorrect attribution creates direct professional liability; users need to click through to verify the exact supporting language themselves.

```
1. Citation granularity: span-level (Q1, Q10) is mandatory here, not just
   preferred -- a legal citation needs to point to the exact holding or
   statutory language, not just "this case is relevant."

2. Generation (Q2's structured method): force structured claim -> citation
   mappings via tool_use, with passage_ids restricted to specific spans
   within retrieved case text, not whole-document references.

3. Verification (Q11's tiering, tightened): run NLI on every claim as a
   first pass, but escalate EVERY claim (not just uncertain ones) to
   LLM-judge verification given the liability stakes -- the cost
   trade-off in Q11 favors thoroughness over cost savings here.

4. Multi-passage handling (Q13): legal reasoning frequently synthesizes
   multiple cases/statutes into a single argument; the verification
   pipeline must support compound-claim decomposition rather than
   flagging every synthesized legal argument as unsupported by default.

5. Decision gate (Q16): precision threshold set very high (>=99%) with
   mandatory human attorney review of any claim the verifier doesn't
   confirm with high confidence -- this domain does not tolerate the
   "flag and ship anyway" response to unsupported claims that a lower-
   stakes domain might accept.

6. Audit trail: every verification verdict (including LLM-judge
   reasoning, Q11) is logged and retained, since a legal deployment may
   need to defend its citation methodology after the fact, not just at
   generation time.
```

The key design choice is treating this as a domain where verification thoroughness (checking every claim, not sampling) is worth its cost, in direct contrast to the load-bearing-claim sampling strategy that's appropriate for lower-stakes, higher-volume use cases (Q14) — the right verification intensity is a function of the cost of an undetected error, not a fixed architectural default.

</details>

---

## Q19. What happens when the verifier itself is wrong, and how do you audit verifier quality? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A verifier's own error modes compound in two directions with different consequences: a **false positive** (verifier says "supported" when the claim actually isn't) is the more dangerous failure, since it lets an unsupported or wrong claim through the exact gate designed to catch it, undermining the entire point of the architecture without any visible signal that something went wrong. A **false negative** (verifier says "not supported" when the claim actually is well-grounded) is less dangerous but still costly — it causes correct claims to be needlessly flagged, removed, or trigger unnecessary re-retrieval (Q3's response options), degrading answer completeness and user trust in the system's caution.

**Auditing verifier quality:** treat the verifier itself as a model requiring the same evaluation discipline as any other component — build a held-out set of (claim, passage, human-judged verdict) pairs specifically designed to probe known NLI weak points (negation, numerical precision, implicit/multi-step entailment, Q17), and measure the verifier's precision/recall against human judgment on this set, not just trust that "NLI models are generally reliable." Re-run this audit whenever the verification model version changes, and monitor production for symptoms of drift — a rising rate of claims sitting exactly at the confidence threshold boundary (Q17's gaming signal) is often the first observable sign that verifier calibration has shifted, before any downstream quality metric visibly degrades.

</details>

---

## Q20. What are the limitations of Verifiable/Citation RAG, and how might the field evolve? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Current limitations: (1) **verification confirms internal consistency, not source truthfulness** (Q17) — a citation can be perfectly verified against a source that is itself wrong or fabricated, which is a gap no amount of NLI or LLM-judge sophistication closes on its own; (2) **granularity is a cost-precision trade-off with no free option** (Q10, Q12) — finer granularity (span-level) gives stronger guarantees but costs more to generate and verify, and coarser granularity is cheaper but structurally permits misattribution to slip through; (3) **multi-passage and compound claims require extra handling** (Q13) that a naive single-passage verification pipeline doesn't provide out of the box; (4) **verification cost scales with claim count** (Q14), forcing a sampling/tiering trade-off for long-form outputs that a compliance-critical domain (Q18) may not be able to accept.

Likely evolution: tighter integration between retrieval-time source-trust scoring and generation-time citation verification, so that a "verified" citation carries an explicit confidence signal reflecting *both* internal consistency and source reliability rather than treating them as unrelated concerns; span-level citation becoming cheaper and more standard as models improve at precise extraction (reducing the granularity cost trade-off in Q10); and, as with SURGE's own trajectory (#40, Q20), domain-specific fine-tuned verification models replacing general-purpose NLI models as production experience accumulates evidence about where general-purpose entailment classifiers systematically fail for a given domain's phrasing conventions.

</details>

---

## Real-World Applications

- **Perplexity.ai and Bing Copilot**: Inline citations with hover-to-verify passage display
- **Legal research (Lexis AI, Harvey)**: Every statement in a legal brief must be attributable to a cited case or statute — attribution verification is a compliance requirement
- **Medical literature assistants**: Claims about drug interactions or clinical outcomes must cite specific study passages
- **ALCE benchmark**: Stanford benchmark specifically for evaluating attribution in long-form answers (Gao et al., 2023)
- **Enterprise compliance reporting**: Audit trails require traceability from every output claim back to a source document
