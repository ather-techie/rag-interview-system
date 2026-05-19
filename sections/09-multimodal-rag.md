# 09 — Multi-modal RAG

> Retrieves across multiple data types — text, images, tables, audio — using multi-modal embeddings and vision-language models.

---

## Q1. What is Multi-modal RAG and what data types does it support? `[Basic]`

**Answer:**

**Multi-modal RAG** extends RAG beyond plain text to support retrieval and generation across heterogeneous data types:

| Modality | Examples | Embedding Model |
|---|---|---|
| Text | Documents, PDFs, web pages | text-embedding-3, BGE |
| Images | Photos, diagrams, screenshots | CLIP, SigLIP, OpenCLIP |
| Tables | CSV, spreadsheets, HTML tables | Table2Text + text embeddings |
| Audio | Transcripts, voice notes | Whisper → text → text embeddings |
| Video | Keyframes + transcripts | CLIP (frames) + text (transcript) |

The key challenge is **cross-modal retrieval** — a text query should be able to retrieve relevant images, and an image query should retrieve relevant text.

---

## Q2. How does CLIP enable cross-modal retrieval? `[Intermediate]`

**Answer:**

**CLIP (Contrastive Language-Image Pretraining)** by OpenAI jointly trains text and image encoders so that semantically related text-image pairs are close in the same embedding space:

```
"a dog playing fetch" ──[text encoder]──► vector [0.2, 0.8, ...]
[photo of dog with ball] ─[image encoder]► vector [0.21, 0.79, ...]
                                             ↑ close in cosine space
```

**For Multi-modal RAG:**
1. At indexing time: embed all images with the CLIP image encoder; embed all text with the CLIP text encoder.
2. At query time: embed the user's text query with the CLIP text encoder.
3. Run ANN search across the combined index — returns both relevant text chunks AND relevant images.
4. Pass top results (images + text) to a vision-language model (GPT-4V, LLaVA, Gemini) for generation.

**Limitation:** CLIP was trained on web images; performance degrades on domain-specific diagrams, medical imaging, or technical schematics without fine-tuning.

---

## Q3. How do you handle tables and structured data in a Multi-modal RAG system? `[Intermediate]`

**Answer:**

Tables require special treatment because raw HTML/CSV doesn't embed well. Common approaches:

**Option 1 — Table-to-text:**
Convert tables to natural language summaries using an LLM:
> "The table shows Q3 revenue by region. North America: $2.1B (+8%), EMEA: $1.4B (+12%)..."
Then embed the summary as text.

**Option 2 — Table as image:**
Render the table as a PNG and embed with CLIP. Useful for complex visual layouts.

**Option 3 — Structured retrieval:**
Don't embed tables at all — route table queries to a Text-to-SQL pipeline that queries the underlying database directly.

**Option 4 — Hybrid:**
Embed the table header + column names for retrieval. At generation time, pass the full raw table to the LLM in the context.

**Best practice:** Use Option 3 for large, queryable datasets and Option 1 for small reference tables in documents.

---

## Q4. What are the key challenges in building a production Multi-modal RAG system? `[Advanced]`

**Answer:**

| Challenge | Description | Solution |
|---|---|---|
| **Index heterogeneity** | Text, image, table embeddings live in different spaces | Use a shared embedding space (CLIP) or separate indexes with a fusion layer |
| **Chunking images** | Images don't chunk like text | Extract regions of interest (ROI) or use the full image |
| **PDF parsing** | PDFs mix text, tables, figures | Use specialized parsers (Unstructured.io, Azure Document Intelligence) |
| **VLM context limits** | Can't pass 20 images to GPT-4V | Rerank to top 2–3 images before generation |
| **Latency** | Image encoding is slower than text | Pre-compute image embeddings offline; use caching |
| **Evaluation** | Hard to evaluate image-grounded answers | Use VQA benchmarks; human eval for production |

---

## Q5. How would you architect a Multi-modal RAG system for a technical documentation chatbot that includes diagrams and code? `[Advanced]`

**Answer:**

**Ingestion pipeline:**
```
PDF/HTML docs
  ├── Text blocks    → text-embedding-3 → Pinecone (text index)
  ├── Code blocks    → CodeBERT / text-embedding → Pinecone (code index)
  ├── Diagrams/figs  → CLIP image encoder → Pinecone (image index)
  └── Tables         → LLM → text summary → Pinecone (text index)
```

**Query pipeline:**
```
User query
  ├── Text retrieval    → top-5 text chunks
  ├── Image retrieval   → top-2 diagrams (CLIP text encoder query)
  └── Code retrieval    → top-3 code snippets
          │
          ▼
     Reranker (filter irrelevant results)
          │
          ▼
  GPT-4V / Gemini 1.5 Pro (with images inline in prompt)
          │
          ▼
     Answer with text + referenced diagrams
```

**Key choices:**
- Use **GPT-4V or Gemini** — they handle interleaved text and image inputs natively.
- Store **original image URLs** alongside embeddings so the UI can display the source diagram.
- Build a **citation layer** — each part of the answer should cite which chunk or diagram it came from.
- Use **Unstructured.io** or **LlamaParse** for robust PDF parsing that preserves layout context.
