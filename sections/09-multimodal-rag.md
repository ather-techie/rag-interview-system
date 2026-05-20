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

---

## Q6. How do you index and retrieve video content in a multimodal RAG system? `[Intermediate]`

**Answer:**

Video requires decomposition into frames and transcripts before embedding and retrieval.

```
Video file (.mp4, .mov, etc.)
    │
    ├─ [Decompose into frames]
    │  ├─ Sample frames every N seconds (e.g., every 5s)
    │  └─ Embed frame-N with CLIP image encoder
    │      → Stored in image index with timestamp metadata
    │
    ├─ [Extract transcript]
    │  ├─ Whisper: speech-to-text
    │  └─ Embed transcript chunks with text-embedding-3
    │      → Stored in text index with timestamp metadata
    │
    └─ [Link frames to transcript]
       Join frame embeddings with corresponding transcript chunks
       (e.g., frame at 2:30s links to transcript "00:02:30 - ...user clicks...")
```

**Retrieval:**

```python
class VideoRAG:
    def index_video(self, video_path: str):
        """Index a video for multi-modal retrieval."""
        
        # Extract frames
        frames = extract_frames_every_n_seconds(video_path, interval=5)
        frame_embeddings = [clip_image_encoder(f) for f in frames]
        
        # Extract audio and transcribe
        audio = extract_audio(video_path)
        transcript = whisper.transcribe(audio)  # List of (timestamp, text) pairs
        
        # Embed transcript
        transcript_chunks = [chunk for ts, chunk in transcript]
        transcript_embeddings = [text_embedding_model(chunk) for chunk in transcript_chunks]
        
        # Store with cross-references
        for i, (frame_emb, frame) in enumerate(zip(frame_embeddings, frames)):
            timestamp = i * 5  # seconds
            self.vectorstore.add(
                embedding=frame_emb,
                metadata={"type": "video_frame", "video": video_path, "timestamp": timestamp},
                doc=frame
            )
        
        for ts, (trans_emb, text) in enumerate(zip(transcript_embeddings, transcript_chunks)):
            self.vectorstore.add(
                embedding=trans_emb,
                metadata={"type": "video_transcript", "video": video_path, "timestamp": ts},
                doc=text
            )
    
    def retrieve_video_content(self, query: str, k=5):
        """Retrieve both frames and transcript for a query."""
        
        # Text query → search both text and image indices
        text_results = self.vectorstore.similarity_search(query, k=k)
        
        # Get frames at same timestamps
        frame_results = []
        for result in text_results:
            if result.metadata["type"] == "video_transcript":
                timestamp = result.metadata["timestamp"]
                # Find corresponding frame
                frame = get_frame_at_timestamp(timestamp)
                frame_results.append({
                    "text": result.page_content,
                    "image": frame,
                    "timestamp": timestamp
                })
        
        return frame_results

# Usage
video_rag = VideoRAG()
video_rag.index_video("tutorial.mp4")
results = video_rag.retrieve_video_content("How do I set up the system?")
# Returns: [{"text": "First, open...", "image": <PIL image>, "timestamp": 12}, ...]
```

---

## Q7. What is early fusion vs late fusion for multimodal retrieval? `[Intermediate]`

**Answer:**

Two strategies for combining multi-modal data in retrieval:

**Early Fusion — Combine before retrieval:**
```
Text embedding     Image embedding
      ↓                    ↓
      └─ Concatenate ─→ Single combined embedding
                             │
                             ▼
                      ANN search (single index)
                             │
                             ▼
                    Top-k results (mixed types)
```

**Late Fusion — Retrieve separately, merge results:**
```
Text query          Image query (from image)
      ↓                    ↓
Text retrieval      Image retrieval
(text index)       (image index)
      ↓                    ↓
 Top-k text        Top-k images
      └──────┬─────────────┘
             ▼
      RRF or learned ranking
             ▼
     Final merged top-k
```

**Comparison:**

| Aspect | Early Fusion | Late Fusion |
|--------|-------------|-----------|
| **Index size** | Single large index | Multiple smaller indexes |
| **Query latency** | Single ANN search | Parallel searches, then merge |
| **Flexibility** | Hard to add new modalities | Easy to add new modalities |
| **Cross-modal search** | Native (embeddings share space) | Requires cross-modal bridge |
| **Ranking control** | Fixed at indexing time | Tunable at query time (RRF weights) |

**Early fusion example (CLIP-based):**
```python
# All modalities embed to same CLIP space
text_embedding = clip_text_encoder("What is this diagram?")
image_embedding = clip_image_encoder(diagram_image)
# Embeddings are comparable; single ANN index
```

**Late fusion example (FAISS hybrid):**
```python
# Separate indexes
text_results = faiss_text_index.search(query_embedding, k=10)
image_results = faiss_image_index.search(clip_image_query, k=10)
# Merge via RRF
merged = reciprocal_rank_fusion([text_results, image_results], weights=[0.6, 0.4])
```

**Recommendation:**
- Use **early fusion** when modalities are naturally comparable (text + images in CLIP space).
- Use **late fusion** for heterogeneous data (structured tables + unstructured text + images).

---

## Q8. How do you parse and embed PDFs containing mixed text, tables, and images? `[Advanced]`

**Answer:**

PDFs are notoriously tricky; layout, fonts, and embedded objects break naive text extraction.

```python
from unstructured.partition.pdf import partition_pdf
from PIL import Image
import io

class PDFMultimodalParser:
    def parse_pdf(self, pdf_path: str) -> dict:
        """Extract text, tables, and images from PDF."""
        
        # Use Unstructured.io to preserve layout
        elements = partition_pdf(
            pdf_path,
            extract_images_in_pdf=True,  # Extract images
            infer_table_structure=True,  # Parse table structure
            strategy="hi_res",  # Use model-based layout analysis (slower but better)
        )
        
        parsed = {
            "text_chunks": [],
            "tables": [],
            "images": []
        }
        
        for element in elements:
            # Classify by element type
            if element.category == "Table":
                # Tables: convert to text summary + keep raw for SQL queries
                table_summary = self.summarize_table(element.text)
                parsed["tables"].append({
                    "raw_html": element.text,
                    "summary": table_summary,
                    "metadata": {"page": element.metadata.page_number}
                })
            
            elif element.category == "Figure" or element.category == "Image":
                # Images: extract and store
                image_data = element.metadata.image_base64  # If available
                if image_data:
                    parsed["images"].append({
                        "base64": image_data,
                        "metadata": {"page": element.metadata.page_number}
                    })
            
            else:  # Text, heading, etc.
                parsed["text_chunks"].append({
                    "text": element.text,
                    "type": element.category,
                    "metadata": {"page": element.metadata.page_number}
                })
        
        return parsed
    
    def summarize_table(self, table_html: str) -> str:
        """Convert table to natural language summary."""
        
        prompt = f"""Summarize this table in 1-2 sentences:
{table_html}

Summary:"""
        
        summary = llm.invoke(prompt)
        return summary.content
    
    def embed_multimodal_pdf(self, pdf_path: str, vectorstore):
        """Parse, embed, and index PDF elements."""
        
        parsed = self.parse_pdf(pdf_path)
        
        # Embed text chunks
        for chunk in parsed["text_chunks"]:
            embedding = text_embedding_model.embed(chunk["text"])
            vectorstore.add(
                embedding=embedding,
                metadata={**chunk["metadata"], "type": "text", "source": pdf_path},
                text=chunk["text"]
            )
        
        # Embed table summaries
        for table in parsed["tables"]:
            embedding = text_embedding_model.embed(table["summary"])
            vectorstore.add(
                embedding=embedding,
                metadata={**table["metadata"], "type": "table", "source": pdf_path, "raw_html": table["raw_html"]},
                text=table["summary"]
            )
        
        # Embed images
        for image in parsed["images"]:
            image_obj = Image.open(io.BytesIO(base64.b64decode(image["base64"])))
            embedding = clip_image_encoder(image_obj)
            vectorstore.add(
                embedding=embedding,
                metadata={**image["metadata"], "type": "image", "source": pdf_path},
                image=image_obj
            )
        
        print(f"Indexed {len(parsed['text_chunks'])} text, {len(parsed['tables'])} tables, {len(parsed['images'])} images")
```

---

## Q9. What evaluation frameworks exist for multimodal RAG? `[Advanced]`

**Answer:**

Evaluating multi-modal RAG is harder than text-only because answers span images, text, and structured data.

**Metric taxonomy:**

| Category | Metrics | Tools |
|----------|---------|-------|
| **Visual QA** | Accuracy on image captioning/VQA | NVSA, MSCOCO Captions, Flickr30K |
| **Cross-modal retrieval** | Recall@k for image-text pairs | MSCOCO (text→image), Flickr (image→text) |
| **Table QA** | Exact match, F1 on table queries | WikiTableQuestions, SPIDER (SQL) |
| **Citation accuracy** | % of claims cited to source (text/image/table) | Custom annotation |
| **Semantic consistency** | Does answer match all retrieved modalities? | LLM-based consistency scoring |

```python
from torchmetrics.retrieval import RetrievalRecall
import numpy as np

class MultimodalRAGEvaluator:
    def __init__(self, test_set):
        # test_set: [(query, gold_text, gold_images, gold_table)]
        self.test_set = test_set
    
    def evaluate_cross_modal_recall(self, system):
        """Can text queries retrieve relevant images?"""
        
        recalls = []
        for query, _, gold_images, _ in self.test_set:
            # System retrieves
            retrieved = system.retrieve(query)
            
            # Check if any retrieved image is in gold_images
            retrieved_images = [r for r in retrieved if r["type"] == "image"]
            gold_image_ids = set(img["id"] for img in gold_images)
            
            hit = any(img["id"] in gold_image_ids for img in retrieved_images)
            recalls.append(float(hit))
        
        return np.mean(recalls)
    
    def evaluate_citation_accuracy(self, system):
        """Are claims properly cited to their sources?"""
        
        correct_citations = 0
        total_claims = 0
        
        for query, _, _, _ in self.test_set:
            answer = system.answer(query)  # Generates answer + cites sources
            
            # Parse citations from answer
            # E.g., "Revenue was $10M [Text:chunk_5, Image:diagram_2]"
            claims_with_citations = extract_claims_and_citations(answer)
            
            for claim, citations in claims_with_citations:
                total_claims += 1
                
                # Verify each citation is valid
                all_valid = all(
                    verify_claim_in_source(claim, citation)
                    for citation in citations
                )
                
                if all_valid:
                    correct_citations += 1
        
        return correct_citations / total_claims if total_claims > 0 else 0
    
    def evaluate_consistency(self, system):
        """Does answer align with all retrieved modalities?"""
        
        consistency_scores = []
        for query, _, _, _ in self.test_set:
            retrieved = system.retrieve(query)
            answer = system.answer(query)
            
            # LLM judges: "Is the answer consistent with all retrieved text/images/tables?"
            consistency_prompt = f"""Query: {query}
Answer: {answer}

Retrieved text: {[r['text'] for r in retrieved if r['type'] == 'text']}
Retrieved images: [show images]
Retrieved table: [table summary]

Rate consistency (0-1): Are all claims consistent with retrieved content?"""
            
            score = float(llm.invoke(consistency_prompt).content)
            consistency_scores.append(score)
        
        return np.mean(consistency_scores)

# Benchmark datasets
benchmarks = {
    "MSCOCO Captions": ("Image captioning", 5K test images),
    "Flickr30K": ("Image-text retrieval", 1K queries),
    "WikiTableQuestions": ("Table QA", 22K questions),
    "NVSA": ("Visual semantic alignment", 7K image-text pairs),
}
```

---

## Q10. How do you fine-tune CLIP for domain-specific multimodal retrieval? `[Advanced]`

**Answer:**

CLIP is trained on web images; fine-tuning on domain data improves retrieval significantly (5–20pp gains).

```python
import torch
from transformers import CLIPModel, CLIPProcessor
from torch.optim import AdamW

class CLIPDomainFineTuning:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
    
    def prepare_contrastive_dataset(self, domain_image_text_pairs):
        """
        Args:
            domain_image_text_pairs: List of (image_path, matching_text_captions)
        
        For each image, create positive (image, caption) pairs and 
        negative pairs from other images.
        """
        
        training_data = []
        
        for i, (image_path, captions) in enumerate(domain_image_text_pairs):
            image = Image.open(image_path)
            
            # Positive pairs: this image with all its captions
            for caption in captions:
                training_data.append({
                    "image": image,
                    "text": caption,
                    "label": 1  # positive
                })
            
            # Negative pairs: this image with captions from other images
            for j in range(max(0, i-2), min(len(domain_image_text_pairs), i+3)):
                if i != j:
                    _, other_captions = domain_image_text_pairs[j]
                    for caption in other_captions[:1]:  # Limit negatives
                        training_data.append({
                            "image": image,
                            "text": caption,
                            "label": 0  # negative
                        })
        
        return training_data
    
    def fine_tune(self, training_data, num_epochs=3, batch_size=32):
        """Contrastive loss fine-tuning."""
        
        from torch.utils.data import DataLoader, Dataset
        
        class ContrastiveDataset(Dataset):
            def __init__(self, data, processor):
                self.data = data
                self.processor = processor
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                example = self.data[idx]
                
                image_inputs = self.processor(
                    images=example["image"],
                    return_tensors="pt"
                )
                text_inputs = self.processor(
                    text=example["text"],
                    return_tensors="pt"
                )
                
                return {
                    "image_pixel_values": image_inputs["pixel_values"].squeeze(),
                    "text_input_ids": text_inputs["input_ids"].squeeze(),
                    "label": example["label"]
                }
        
        dataset = ContrastiveDataset(training_data, self.processor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for batch in loader:
                # Forward pass
                image_outputs = self.model.vision_model(
                    pixel_values=batch["image_pixel_values"]
                )
                text_outputs = self.model.text_model(
                    input_ids=batch["text_input_ids"]
                )
                
                # Project to embedding space
                image_embeds = self.model.visual_projection(image_outputs.pooler_output)
                text_embeds = self.model.text_projection(text_outputs.pooler_output)
                
                # Contrastive loss (simplified)
                batch_labels = batch["label"]
                logits = torch.matmul(image_embeds, text_embeds.T) / 0.07
                loss = torch.nn.functional.cross_entropy(logits, batch_labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch}, Loss: {total_loss / len(loader):.4f}")
    
    def evaluate_domain_performance(self, domain_test_pairs):
        """Measure improvement on domain-specific queries."""
        
        recalls = []
        
        for image, captions in domain_test_pairs:
            # Encode query image
            image_inputs = self.processor(images=image, return_tensors="pt")
            image_embeds = self.model.vision_model(**image_inputs).pooler_output
            image_embeds = self.model.visual_projection(image_embeds)
            
            # Encode text options
            text_inputs = self.processor(text=captions, return_tensors="pt", padding=True)
            text_embeds = self.model.text_model(**text_inputs).pooler_output
            text_embeds = self.model.text_projection(text_embeds)
            
            # Rank: top-k captions should be the correct ones
            logits = torch.matmul(image_embeds, text_embeds.T)
            top_k_indices = torch.argsort(logits[0], descending=True)[:5]
            
            # Recall@5: first gold caption in top 5?
            recall = 1.0 if 0 in top_k_indices else 0.0
            recalls.append(recall)
        
        return np.mean(recalls)

# Pitfalls and mitigation
pitfalls = {
    "Overfitting to domain": {
        "problem": "Model memorizes training pairs, fails on new images",
        "solution": "Data augmentation (crop, rotate), validation set monitoring"
    },
    "Catastrophic forgetting": {
        "problem": "Fine-tuning on domain data destroys general capabilities",
        "solution": "Use LoRA (low-rank adaptation) instead of full fine-tuning; mix general + domain data"
    },
    "Class imbalance": {
        "problem": "Some domain categories have few examples",
        "solution": "Weighted sampling, contrastive hard-negative mining"
    }
}

# Typical results:
# ├─ Base CLIP on domain: 45% recall@5
# ├─ + Fine-tune 5K domain pairs: 72% recall@5  (+27pp)
# └─ + LoRA (20 adapters): 74% recall@5, retains 98% general performance
```

**Recommendation:**
Use **LoRA fine-tuning** over full fine-tuning to avoid forgetting. Target 5–10K domain image-caption pairs for a 15–20pp gain.
