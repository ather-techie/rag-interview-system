# References

Full citations with links for every paper referenced in this repo, grouped by topic. Each entry includes a one-line note on why (and when) to read it.

> Reading strategy: you don't need to read papers to pass most RAG interviews — the Q&A files summarize what matters. Read the **Foundations** papers if you're interviewing for research-adjacent roles; skim the architecture papers for whichever RAG types your target role emphasizes.

---

## Foundations

| Paper | Link | Why read it |
|---|---|---|
| Lewis et al., 2020 — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* | [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) | The original RAG paper; defines retrieve-then-generate. Used in [`02_interview_bank/01-naive-rag.md`](./02_interview_bank/01-naive-rag.md) |
| Karpukhin et al., 2020 — *Dense Passage Retrieval for Open-Domain Question Answering* | [arXiv:2004.04906](https://arxiv.org/abs/2004.04906) | DPR — the bi-encoder dense retrieval recipe everything else builds on |
| Reimers & Gurevych, 2019 — *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks* | [arXiv:1908.10084](https://arxiv.org/abs/1908.10084) | Why bi-encoders exist; the sentence-transformers library behind most embedding fine-tuning |
| Gao et al., 2023 — *Retrieval-Augmented Generation for Large Language Models: A Survey* | [arXiv:2312.10997](https://arxiv.org/abs/2312.10997) | The survey that popularized the Naive → Advanced → Modular taxonomy used in this repo |

## Architecture papers (by interview bank section)

| Paper | Link | Bank section |
|---|---|---|
| Yao et al., 2022 — *ReAct: Synergizing Reasoning and Acting in Language Models* | [arXiv:2210.03629](https://arxiv.org/abs/2210.03629) | [`04-agentic-rag.md`](./02_interview_bank/04-agentic-rag.md) — the reason/act loop agentic RAG is built on |
| Jiang et al., 2023 — *Active Retrieval Augmented Generation* (FLARE) | [arXiv:2305.06983](https://arxiv.org/abs/2305.06983) | [`04-agentic-rag.md`](./02_interview_bank/04-agentic-rag.md) — generate, detect uncertainty, re-retrieve |
| Edge et al., 2024 — *From Local to Global: A Graph RAG Approach to Query-Focused Summarization* | [arXiv:2404.16130](https://arxiv.org/abs/2404.16130) | [`05-graph-rag.md`](./02_interview_bank/05-graph-rag.md) — Microsoft GraphRAG: KG + community detection |
| Yan et al., 2024 — *Corrective Retrieval Augmented Generation* (CRAG) | [arXiv:2401.15884](https://arxiv.org/abs/2401.15884) | [`06-corrective-rag.md`](./02_interview_bank/06-corrective-rag.md) — retrieval evaluator + web-search fallback |
| Asai et al., 2023 — *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection* | [arXiv:2310.11511](https://arxiv.org/abs/2310.11511) | [`07-self-rag.md`](./02_interview_bank/07-self-rag.md) — reflection tokens, trained retrieval decisions |
| Wang et al., 2024 — *Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting* | [arXiv:2407.08223](https://arxiv.org/abs/2407.08223) | [`08-speculative-rag.md`](./02_interview_bank/08-speculative-rag.md) — small drafter + large verifier |
| Radford et al., 2021 — *Learning Transferable Visual Models From Natural Language Supervision* (CLIP) | [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) | [`09-multimodal-rag.md`](./02_interview_bank/09-multimodal-rag.md) — the joint text-image embedding space |
| Faysse et al., 2024 — *ColPali: Efficient Document Retrieval with Vision Language Models* | [arXiv:2407.01449](https://arxiv.org/abs/2407.01449) | [`09-multimodal-rag.md`](./02_interview_bank/09-multimodal-rag.md) — late-interaction retrieval over page images, no OCR |
| Liu et al., 2023 — *Lost in the Middle: How Language Models Use Long Contexts* | [arXiv:2307.03172](https://arxiv.org/abs/2307.03172) | [`10-long-context-rag.md`](./02_interview_bank/10-long-context-rag.md) — why stuffing context fails; U-shaped attention |
| Jiang et al., 2023 — *LLMLingua: Compressing Prompts for Accelerated Inference of LLMs* | [arXiv:2310.05736](https://arxiv.org/abs/2310.05736) | [`10-long-context-rag.md`](./02_interview_bank/10-long-context-rag.md) — prompt compression for token cost |
| Jeong et al., 2024 — *Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity* | [arXiv:2403.14403](https://arxiv.org/abs/2403.14403) | [`11-adaptive-rag.md`](./02_interview_bank/11-adaptive-rag.md) — query-complexity classifier routing |

## Retrieval & indexing

| Paper | Link | Why read it |
|---|---|---|
| Robertson & Zaragoza, 2009 — *The Probabilistic Relevance Framework: BM25 and Beyond* | [DOI:10.1561/1500000019](https://doi.org/10.1561/1500000019) | The definitive BM25 reference behind sparse/hybrid retrieval |
| Formal et al., 2021 — *SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval* | [arXiv:2109.10086](https://arxiv.org/abs/2109.10086) | Learned sparse retrieval — the modern half of hybrid search. See [`01_concepts/retrieval_strategies.md`](./01_concepts/retrieval_strategies.md) |
| Khattab & Zaharia, 2020 — *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction* | [arXiv:2004.12832](https://arxiv.org/abs/2004.12832) | Late interaction — the middle ground between bi- and cross-encoders |
| Gao et al., 2022 — *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE) | [arXiv:2212.10496](https://arxiv.org/abs/2212.10496) | Hypothetical document embeddings for query rewriting. See [`02-advanced-rag.md`](./02_interview_bank/02-advanced-rag.md) |
| Zheng et al., 2023 — *Take a Step Back: Evoking Reasoning via Abstraction in LLMs* | [arXiv:2310.06117](https://arxiv.org/abs/2310.06117) | Step-back prompting for query abstraction. See [`01_concepts/retrieval_strategies.md`](./01_concepts/retrieval_strategies.md) |
| Cormack et al., 2009 — *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods* | [DOI:10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114) | RRF — the standard way to merge dense + sparse result lists |
| Malkov & Yashunin, 2016 — *Efficient and Robust Approximate Nearest Neighbor Search Using HNSW Graphs* | [arXiv:1603.09320](https://arxiv.org/abs/1603.09320) | The index behind most vector DBs. See [`01_concepts/vector_databases.md`](./01_concepts/vector_databases.md) |
| Sun et al., 2023 — *Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents* (RankGPT) | [arXiv:2304.09542](https://arxiv.org/abs/2304.09542) | LLM listwise reranking. See [`01_concepts/reranking.md`](./01_concepts/reranking.md) |
| Kusupati et al., 2022 — *Matryoshka Representation Learning* | [arXiv:2205.13147](https://arxiv.org/abs/2205.13147) | Truncatable embeddings — why `text-embedding-3` lets you cut dimensions. See [`01_concepts/embeddings.md`](./01_concepts/embeddings.md) |
| Wang et al., 2021 — *GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval* | [arXiv:2112.07577](https://arxiv.org/abs/2112.07577) | Synthetic training data for embedding fine-tuning. See [`01_concepts/fine_tuning.md`](./01_concepts/fine_tuning.md) |

## Evaluation & benchmarks

| Paper | Link | Why read it |
|---|---|---|
| Es et al., 2023 — *RAGAS: Automated Evaluation of Retrieval Augmented Generation* | [arXiv:2309.15217](https://arxiv.org/abs/2309.15217) | Reference-free RAG evaluation — the de facto standard. See [`01_concepts/evaluation_metrics.md`](./01_concepts/evaluation_metrics.md) |
| Liu et al., 2023 — *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment* | [arXiv:2303.16634](https://arxiv.org/abs/2303.16634) | LLM-as-judge with chain-of-thought rubrics |
| Thakur et al., 2021 — *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models* | [arXiv:2104.08663](https://arxiv.org/abs/2104.08663) | The retrieval generalization benchmark |
| Muennighoff et al., 2022 — *MTEB: Massive Text Embedding Benchmark* | [arXiv:2210.07316](https://arxiv.org/abs/2210.07316) | How embedding models are compared; read before trusting any leaderboard number |

## Security

| Paper | Link | Why read it |
|---|---|---|
| Greshake et al., 2023 — *Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection* | [arXiv:2302.12173](https://arxiv.org/abs/2302.12173) | Indirect prompt injection — the core RAG attack surface. See [`01_concepts/prompt_injection_risks.md`](./01_concepts/prompt_injection_risks.md) |
| Hines et al., 2024 — *Defending Against Indirect Prompt Injection Attacks With Spotlighting* | [arXiv:2403.14720](https://arxiv.org/abs/2403.14720) | Marking untrusted content so the model can tell instructions from data |
