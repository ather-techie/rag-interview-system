# 35 — Streaming / Real-Time RAG

> How to keep a RAG index continuously up-to-date using event streams and CDC pipelines, reducing the freshness window from hours/days to seconds.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
Batch RAG (standard):
  Documents ──► Batch Pipeline (nightly) ──► Vector Index (stale)

Streaming RAG:
  Documents ──► Event Stream ──► Streaming Pipeline (seconds) ──► Vector Index (fresh)

┌─────────────────────────────────────────────────────────────┐
│  DATA SOURCES                                               │
│  ├─ Database (Postgres CDC / Debezium)                      │
│  ├─ APIs (webhooks, polling)                                │
│  └─ File systems (S3 event notifications)                   │
└─────────────────────────────┬───────────────────────────────┘
                              │  events (create/update/delete)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  EVENT STREAM (Kafka / Kinesis / Pub/Sub)                   │
│  Topic: rag-document-events                                 │
│  Message: {doc_id, operation, content, timestamp}           │
└─────────────────────────────┬───────────────────────────────┘
                              │  consume
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STREAM PROCESSOR (Kafka Consumer / Faust / Bytewax)        │
│  ├─ Parse & validate                                        │
│  ├─ Chunk text                                              │
│  ├─ Embed (batch within window)                             │
│  └─ Upsert / delete vectors                                 │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  VECTOR INDEX (Qdrant / Weaviate / Pinecone)                │
│  Always reflects latest document state                      │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Responsibility |
|---|---|
| CDC / Event Source Connector | Captures create/update/delete events from databases, APIs, or file systems (e.g., Debezium reading the Postgres WAL) |
| Stream Processor | Consumes the event topic, parses/validates payloads, chunks text, and triggers embedding |
| Incremental Embedder | Embeds only the changed chunks (micro-batched) instead of re-embedding the full corpus |
| Live Index Upserter | Applies delete-then-insert or versioned upserts so the vector index never serves stale chunks |
| Freshness-aware Retriever | Tracks per-document freshness lag and can enforce a max-staleness SLO on query results |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| CDC | Debezium, AWS DMS, Postgres logical replication |
| Event streaming | Kafka, AWS Kinesis, Google Pub/Sub, Redis Streams (lighter-weight alternative) |
| Stream processing | Kafka Streams, Apache Flink, Spark Structured Streaming, Faust/Bytewax |
| Vector index (incremental upsert) | Qdrant, Weaviate, Pinecone |

---

## Q1. What is Streaming / Real-Time RAG and what problem does it solve? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Standard RAG indexes are built in batch: documents are processed, embedded, and loaded into the vector DB on a schedule (hourly, nightly). During the interval between batches, new or changed documents are invisible to retrieval — the system confidently answers questions about an outdated world.

Streaming RAG replaces the batch pipeline with a continuous, event-driven pipeline. Every document creation, update, or deletion publishes an event; a stream processor consumes it, re-embeds only the affected chunks, and upserts the vector index immediately, bringing the freshness window down from hours or days to seconds. This matters wherever the value of an answer decays quickly with staleness — news, live pricing, support documentation that changes during an incident, or any system where "the index was last updated last night" is an unacceptable answer to a user.

</details>

---

## Q2. What is the single distinctive mechanism that separates Streaming RAG from standard batch-indexed RAG? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The mechanism is **event-driven incremental indexing in place of scheduled full (or partial) batch runs**. Batch RAG treats indexing as a periodic job that reprocesses a set of documents on a clock; streaming RAG treats every document mutation as a first-class event that triggers its own, isolated indexing step the moment it happens.

Concretely, this means: no cron schedule, no "wait for the next run"; instead a message broker (Kafka/Kinesis) is the system of record for "what changed," and a long-running consumer processes that queue continuously. The corollary is that streaming RAG needs machinery batch RAG doesn't — ordering guarantees, dead-letter queues, delete-before-upsert semantics on the same document — because processing now happens per-event instead of per-batch-run, and a single bad event can no longer be caught by re-running a job; it has to be handled by the pipeline itself.

</details>

---

## Q3. Walk through the end-to-end streaming RAG architecture, from a document change to a fresh query result. `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

```
DATA SOURCES (DB via CDC, APIs via webhook, files via S3 events)
        │  publishes create/update/delete events
        ▼
EVENT STREAM (Kafka topic: rag-document-events)
        │  {doc_id, operation, content, timestamp}
        ▼
STREAM PROCESSOR
        │  parse → chunk → embed (micro-batched) → upsert/delete
        ▼
VECTOR INDEX (Qdrant / Weaviate / Pinecone)
        │  always reflects the latest committed document state
        ▼
QUERY TIME: retrieval sees the update within seconds of the source change
```

Each stage has one job: the data source layer's only responsibility is emitting a reliable event for every mutation (this is where CDC tooling like Debezium plugs in, see Q5); the stream is the durable, ordered buffer between production and consumption so the indexer can fall behind and catch up without losing events; the stream processor does the actual chunk/embed/upsert work, ideally micro-batched (Q8) rather than one document at a time; and the vector index is deliberately kept "dumb" — it just serves whatever is currently upserted, with no awareness that a streaming pipeline feeds it. Freshness (Q11) is measured end-to-end, from `source_updated_at` in the original system to `indexed_at` in the vector DB.

</details>

---

## Q4. How do you implement the event producer and stream consumer for streaming RAG indexing? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The document-side producer publishes one event per mutation:

```python
from kafka import KafkaProducer
import json, time

producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

def publish_document_event(doc_id: str, operation: str, content: str = None):
    """operation: "create" | "update" | "delete" """
    event = {"doc_id": doc_id, "operation": operation, "content": content, "timestamp": time.time()}
    producer.send("rag-document-events", event)
    producer.flush()

publish_document_event("doc:12345", "update", "New policy effective 2026-07-01...")
publish_document_event("doc:9999", "delete")
```

The indexing-side consumer processes each event, deleting-then-reinserting on updates so no stale chunk from the prior version survives:

```python
def process_event(event: dict, vector_db):
    doc_id, operation = event["doc_id"], event["operation"]
    if operation == "delete":
        vector_db.delete(filter={"doc_id": {"$eq": doc_id}})
        return
    if operation in ("create", "update"):
        if operation == "update":
            vector_db.delete(filter={"doc_id": {"$eq": doc_id}})
        chunks = chunk_text(event["content"])
        embeddings = EMBED_MODEL.encode(chunks, normalize_embeddings=True, batch_size=32)
        vectors = [{"id": f"{doc_id}:chunk:{i}", "values": emb.tolist(),
                    "metadata": {"doc_id": doc_id, "chunk_idx": i, "text": c, "timestamp": event["timestamp"]}}
                   for i, (c, emb) in enumerate(zip(chunks, embeddings))]
        vector_db.upsert(vectors=vectors)

def run_indexing_consumer(vector_db):
    consumer = KafkaConsumer("rag-document-events", bootstrap_servers=["localhost:9092"],
                              value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                              group_id="rag-indexing-group", auto_offset_reset="earliest")
    for message in consumer:
        try:
            process_event(message.value, vector_db)
        except Exception as e:
            print(f"Error processing {message.value['doc_id']}: {e}")  # route to a DLQ in production
```

The `group_id` is what makes this horizontally scalable: multiple consumer processes with the same group id share partitions of the topic, each processing a disjoint subset of documents in parallel.

</details>

---

## Q5. How does Change Data Capture (CDC) with Debezium work, and why is it the zero-code option for database-backed content? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

CDC captures row-level database mutations (INSERT/UPDATE/DELETE) and publishes them as a stream of events without requiring any application code changes. Debezium connects to a database's replication log (the write-ahead log, or WAL, in Postgres) and reads every committed change directly from the storage engine's own change feed:

```yaml
name: postgres-rag-connector
config:
  connector.class: io.debezium.connector.postgresql.PostgresConnector
  database.hostname: postgres
  database.dbname: content_db
  table.include.list: public.documents
  transforms: ExtractNewDocumentState
  transforms.ExtractNewDocumentState.type: io.debezium.transforms.ExtractNewRecordState
  topic.prefix: cdc
```

Debezium publishes a message to `cdc.public.documents` for every `INSERT`, `UPDATE`, or `DELETE` on that table. The RAG indexer subscribes to this topic exactly like it would to an application-published event topic (Q4) — the difference is entirely upstream: no application code has to remember to call `publish_document_event()`, because the database itself is the source of truth for what changed. This is the preferred approach whenever documents live in a relational database, since it eliminates an entire class of bugs where an application code path updates a document but forgets to emit the corresponding event. Debezium supports Postgres, MySQL, MongoDB, SQL Server, and Oracle.

</details>

---

## Q6. How does Streaming RAG differ from standard batch-indexed RAG? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

In batch RAG, documents are ingested on a schedule (hourly, nightly), so the index always lags the source of truth by at least one batch interval — a document changed one minute after the last batch run is invisible to retrieval for up to the entire interval. Streaming RAG replaces the batch job with a continuous event-driven pipeline: every mutation publishes an event, and a stream processor re-embeds and upserts the affected chunks within seconds.

The trade-off is infrastructure complexity: streaming requires a message broker, a stream processor, dead-letter queues, and ordering guarantees that a simple batch cron job never needs (Q13, Q14). The payoff is freshness: a query issued moments after a document update sees the update immediately rather than waiting for the next scheduled run. The right choice depends entirely on how fast staleness costs you something — a slowly-changing policy corpus tolerates batch RAG fine; live news or incident-response documentation does not.

</details>

---

## Q7. When would you choose CDC over application-level event publishing as your streaming event source, or vice versa? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Situation | Preferred source |
|---|---|
| Documents live in a relational database you control | CDC (Debezium) — zero application code changes, captures every mutation including ones made by scripts, migrations, or admin tools that would otherwise bypass an application-level publish call |
| Documents come from third-party APIs or SaaS tools (support ticket systems, CMS platforms) | Application-level webhook / polling — there is no database WAL to tap into; you consume whatever change-notification mechanism the source system exposes |
| Documents are files in object storage (S3, GCS) | Native storage event notifications (S3 event notifications) — closest analog to CDC for a non-database source |
| You need guaranteed completeness (every mutation captured, no exceptions) | CDC, because it reads the same log the database itself uses for replication — nothing can mutate data without going through it |
| You need custom business-logic filtering before publishing (e.g., only publish "approved" documents) | Application-level publishing, since CDC gives you raw row changes with no application context |

The general rule: CDC is stronger whenever you own the database, because it cannot be silently bypassed the way a "remember to call publish()" code path can. Application-level publishing becomes necessary the moment the source of truth isn't a database you can attach a log reader to, or when you need semantic filtering that only application code understands.

</details>

---

## Q8. How do you implement micro-batching for embedding efficiency in a streaming indexer? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Calling the embedding model once per document wastes throughput — embedding models amortize much better over batches. A micro-batch indexer collects events for a short window (or until a size threshold) before issuing one embedding call for the whole batch:

```python
import asyncio

class MicroBatchIndexer:
    def __init__(self, vector_db, batch_size: int = 32, flush_interval: float = 0.5):
        self.vector_db, self.batch_size, self.flush_interval = vector_db, batch_size, flush_interval
        self.pending, self._lock = [], asyncio.Lock()

    async def enqueue(self, event: dict):
        async with self._lock:
            self.pending.append(event)
            if len(self.pending) >= self.batch_size:
                await self._flush()

    async def _flush(self):
        if not self.pending:
            return
        batch, self.pending = self.pending[:], []
        all_chunks, all_meta = [], []
        for event in batch:
            if event["operation"] == "delete":
                self.vector_db.delete(filter={"doc_id": {"$eq": event["doc_id"]}})
                continue
            if event["operation"] == "update":
                self.vector_db.delete(filter={"doc_id": {"$eq": event["doc_id"]}})
            for i, chunk in enumerate(chunk_text(event.get("content", ""))):
                all_chunks.append(chunk)
                all_meta.append({"doc_id": event["doc_id"], "chunk_idx": i, "text": chunk})
        if not all_chunks:
            return
        embeddings = EMBED_MODEL.encode(all_chunks, normalize_embeddings=True, batch_size=64)
        vectors = [{"id": f"{m['doc_id']}:chunk:{m['chunk_idx']}", "values": e.tolist(), "metadata": m}
                   for m, e in zip(all_meta, embeddings)]
        self.vector_db.upsert(vectors=vectors)

    async def periodic_flush(self):
        while True:
            await asyncio.sleep(self.flush_interval)
            async with self._lock:
                await self._flush()
```

`flush_interval` is the direct dial on the freshness-vs-throughput trade-off: a 0.5s window adds at most 500ms of extra latency per event but lets embedding calls batch dozens of chunks together, which on GPU-backed embedding endpoints is the difference between saturating the batch dimension and paying per-call overhead on every single chunk.

</details>

---

## Q9. How do you handle document updates in a streaming index without creating duplicate or stale chunks? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Each document's chunks are identified by a `doc_id` metadata field. On an update event: (1) delete all vectors with `doc_id = X` from the index; (2) re-chunk and re-embed the new content; (3) upsert the new vectors. This guarantees no stale chunk from the prior version survives alongside the new one.

The risk is partial failure: if the process crashes after step (1) but before step (3), the document is temporarily invisible to retrieval. Two mitigations are used in practice:

- **Two-phase swap** — write new vectors with a `pending` flag, atomically flip the flag to `active`, then delete the old vectors — so there is never a window where the document has zero active vectors (this is the safer default; see Q14 for the full failure-recovery version).
- **Version-timestamp filtering** — skip the explicit delete step entirely; write new chunks tagged with a `version_ts`, and have queries filter to only the highest `version_ts` per `doc_id`. This accepts a small amount of index bloat (old versions linger until a periodic compaction) in exchange for updates that can never leave the document momentarily invisible.

Delete-then-upsert (the simple version) is fine when a few seconds of document invisibility during an update is acceptable; the two variants above exist specifically for when it is not.

</details>

---

## Q10. What are the key tuning knobs for a streaming RAG pipeline, and how do you choose them? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Knob | Effect | Starting point |
|---|---|---|
| `batch_size` / `flush_interval` (micro-batching) | Trade-off between embedding throughput and per-event latency | 32 events or 0.5s, whichever comes first |
| Consumer group parallelism (number of consumer processes) | Throughput scales roughly linearly with partition count up to the topic's partition count | Start at partition count of the Kafka topic; over-provisioning beyond that is wasted |
| Freshness SLO threshold | Defines what "fresh enough" means and drives alerting | 30 seconds p95 is a common target for "near-real-time" without requiring sub-second engineering |
| Dead-letter queue retry policy (max retries, backoff) | Controls how long a poison event is retried before being quarantined | 3 retries with exponential backoff, then DLQ |
| CDC vs. application-event source | Determines completeness guarantees (Q7) | CDC by default whenever the source is a database you control |

Freshness SLO and micro-batch window interact directly: a tighter freshness target forces a shorter `flush_interval`, which reduces the embedding batch size the pipeline can accumulate before it must flush, which raises embedding cost per document. There is no knob that improves both simultaneously — treat freshness SLO as the business requirement to solve for, and size the other knobs to meet it at the lowest infrastructure cost.

</details>

---

## Q11. How do you measure and monitor freshness lag as a first-class metric? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Freshness lag is the gap between when a document changed in the source system and when the vector index reflects that change — it should be tracked per-event, not just eyeballed:

```python
from dataclasses import dataclass

@dataclass
class FreshnessMetrics:
    doc_id: str
    source_updated_at: float
    indexed_at: float

    @property
    def freshness_lag_seconds(self) -> float:
        return self.indexed_at - self.source_updated_at

FRESHNESS_SLO_SECONDS = 30

def check_freshness_slo(metrics: list[FreshnessMetrics]) -> bool:
    lags = sorted(m.freshness_lag_seconds for m in metrics)
    p95 = lags[int(len(lags) * 0.95)]
    return p95 <= FRESHNESS_SLO_SECONDS
```

Emit `freshness_lag_seconds` as a metric on every processed event (not just a sample), so p50/p95/p99 dashboards are available in real time, and alert specifically on p95/p99 rather than the mean — a mean can look healthy while a subset of documents (e.g., ones hitting the DLQ, Q13) sit stale for minutes. Freshness is the one metric unique to streaming RAG that batch RAG never needs to track at all, which is exactly why treating it as an SLO with dashboards and alerts, not an assumption, matters.

</details>

---

## Q12. How would you build a decision-gate benchmark to validate a streaming pipeline meets its freshness SLO under load? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A load test that only checks throughput can pass while freshness silently degrades, because a pipeline can keep up with steady-state volume while falling permanently behind during bursts. The gate needs to specifically probe burst behavior:

```
1. Baseline: replay a recorded 1-hour window of production event volume at 1x speed;
   record p50/p95/p99 freshness_lag_seconds. This must already meet the SLO — if not,
   nothing below matters yet.

2. Burst test: replay the same window at 5x speed (or inject a synthetic burst of
   N events in a single second, N = your worst observed real burst). Measure whether
   freshness lag recovers to baseline within a defined recovery window (e.g., 2 minutes)
   after the burst ends, not just whether it stays "acceptable" during the burst itself.

3. Failure-injection test: kill one consumer process mid-run (simulating a pod restart)
   and measure how long freshness lag stays elevated before consumer-group rebalancing
   restores full throughput.

4. Gate: FAIL the deployment if (a) baseline p95 exceeds SLO, (b) burst-recovery time
   exceeds the recovery window, or (c) failure-injection recovery exceeds a separate
   (looser) threshold. Record the actual measured numbers in the deployment manifest
   so a regression is visible in the next run's diff, not just a pass/fail bit.
```

The key insight this gate encodes: freshness SLOs are trivially met at steady state by almost any pipeline, so a gate that only tests steady state gives false confidence. The real risk is burst absorption and recovery time, which is exactly what production traffic will eventually exercise.

</details>

---

## Q13. What is the characteristic failure mode of out-of-order events, and how do you handle it? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Events can arrive out of order — network delays, Kafka partition rebalancing, or retried messages from a DLQ can cause an older version of a document to be processed *after* a newer one. Without protection, this silently reverts the index to a stale version of the document with no error raised.

The fix is a timestamp-based check against the last successfully indexed version before applying any event:

```python
from functools import lru_cache

@lru_cache(maxsize=10_000)
def get_last_indexed_timestamp(doc_id: str) -> float:
    result = vector_db.fetch(ids=[f"{doc_id}:chunk:0"])
    if result and result["vectors"]:
        return result["vectors"][f"{doc_id}:chunk:0"]["metadata"].get("timestamp", 0)
    return 0

def should_process(event: dict) -> bool:
    last_ts = get_last_indexed_timestamp(event["doc_id"])
    return event["timestamp"] > last_ts  # skip stale events
```

This makes processing effectively idempotent with respect to ordering: an event older than what's already indexed is simply dropped rather than applied. The `lru_cache` here is a simplification for illustration — in a multi-consumer deployment, the timestamp check needs to hit a shared store (the vector DB itself, or a small side table), not a per-process in-memory cache, or two consumers can each believe they have the latest timestamp while actually racing each other.

</details>

---

## Q14. What happens when the stream processor crashes mid-update, and how do you make updates atomic? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The naive delete-then-upsert sequence (Q9) has a crash window: if the process dies after deleting the old vectors but before the new ones are written, the document is fully absent from the index until the event is reprocessed — and if the consumer's offset was already committed, it may never be reprocessed at all.

Two production-grade fixes:

1. **Two-phase swap.** Write the new chunk vectors tagged `status: pending` alongside the still-active old vectors (no delete yet). Once all new vectors are confirmed written, atomically flip a document-level pointer (a single metadata record, or a separate "current version" key) from the old version's ID prefix to the new one. Only after the swap is confirmed do you delete the old vectors as a cleanup step. At every point in this sequence, a full valid version of the document is queryable — the crash window is eliminated because the "delete" step is now last and non-critical (a crash there just leaves harmless orphaned old vectors for a later cleanup pass).
2. **Idempotent versioned upsert.** Never delete synchronously at all: tag every chunk with `version_ts`, and make the query path filter to `MAX(version_ts)` per `doc_id` (Q9's second variant). Old versions become garbage collected by a periodic background compaction rather than an inline delete, which removes the crash window by removing the delete from the critical path entirely.

Both approaches share the same principle: never let "old data deleted" and "new data present" be two separate, non-atomic steps in the critical path. Commit the Kafka offset only after the vector DB write is confirmed (not before), so a crash mid-processing causes the event to be redelivered and retried rather than silently lost — combined with `should_process` timestamp filtering (Q13), a redelivered event that was actually already applied is safely a no-op.

</details>

---

## Q15. What is the cost and infrastructure overhead of streaming RAG vs. batch RAG at scale? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Dimension | Batch RAG | Streaming RAG |
|-----------|-----------|--------------|
| Freshness | Hours–days | Seconds |
| Infrastructure | Simple (cron job) | Complex (Kafka/Kinesis cluster, stream processor, DLQ) |
| Cost | Low (offline embedding, runs once per interval) | Higher (always-on consumer, real-time embedding calls) |
| Error isolation | Easy (re-run the whole batch) | Harder (partial failures need DLQ + retry + ordering logic) |
| Best for | Stable corpora, low update frequency | News, financial data, live support docs |

Illustrative cost comparison for 1M document updates/day: a nightly batch job amortizes embedding calls into large batches and runs on ephemeral compute that only exists for the job's duration — illustrative cost ≈ $50/day in embedding + compute. A streaming pipeline keeps a consumer process (and, in production, several for parallelism) running 24/7 regardless of event volume, plus a managed Kafka cluster (illustrative $200–500/month baseline) whether or not it's busy — illustrative cost ≈ $80–120/day at the same update volume, dominated by always-on infrastructure rather than the incremental embedding calls themselves, which are actually cheaper per-document than batch since only changed chunks are re-embedded.

The overhead is structural, not just financial: batch RAG's failure mode is "the whole job failed, re-run it," debuggable with a single log; streaming RAG's failure mode is "some subset of events silently failed," requiring DLQ inspection, per-event tracing, and ordering-aware retry logic that batch pipelines simply don't need.

</details>

---

## Q16. How do you scale the stream processor and vector DB upserts as event volume grows? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Scaling a streaming indexer has two independent bottlenecks — consumption throughput and write throughput — and they need separate strategies:

- **Consumer parallelism** — Kafka partitions are the unit of parallelism; a consumer group with N processes can process up to N partitions concurrently, so the topic's partition count sets the throughput ceiling. Partition by `doc_id` hash so all events for the same document land on the same partition (preserving per-document ordering, which `should_process`'s timestamp check depends on) while still parallelizing across documents.
- **Embedding throughput** — micro-batching (Q8) is the primary lever; beyond that, horizontally scale embedding inference (multiple GPU replicas behind a load balancer) rather than trying to push a single instance's batch size arbitrarily high, since batch size has diminishing throughput returns past the point where it saturates the accelerator.
- **Vector DB write throughput** — most vector DBs rate-limit or queue upserts internally; batch upserts (send 100s of vectors per API call, not one) and monitor upsert queue depth as its own metric, since a vector DB write bottleneck manifests identically to a slow embedding step (rising freshness lag) but requires a completely different fix.
- **Backpressure** — when the vector DB or embedding service is the bottleneck, the consumer must slow its Kafka consumption rate rather than buffering unboundedly in memory; most Kafka client libraries support pausing partition consumption, which should be wired to a queue-depth threshold on the downstream write path.

The general principle: identify which of the three stages (consume → embed → write) is actually saturated before scaling any of them — over-provisioning consumer parallelism when the vector DB write path is the bottleneck just produces a bigger in-memory backlog, not more throughput.

</details>

---

## Q17. What security and trust risks does a continuously-updating index introduce, and how do you mitigate them? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Continuous ingestion removes the natural review window that a batch pipeline implicitly provides (someone can eyeball a nightly diff before it goes live); a streaming pipeline makes malicious or erroneous content live within seconds:

- **Rapid poisoning** — an attacker with write access to the source system (or a compromised upstream integration) can push a burst of malicious documents that are indexed and retrievable almost immediately, with no batch window to catch them. Mitigate with the same content-screening techniques used in batch ingestion (profanity/malicious-content filters, trust scoring) applied inline in the stream processor, plus rate-limiting on write volume per source/user to slow a burst attack to a containable pace.
- **Replay attacks** — a stale or previously-deleted event replayed into the stream (e.g., from a misconfigured retry or a compromised producer) can resurrect deleted content or roll back a legitimate update. The `should_process` timestamp check (Q13) is the primary defense, but it depends on trusting the event's `timestamp` field — for adversarial replay resistance, use a monotonic source-of-truth timestamp (the CDC log's own commit LSN, not a client-supplied field) rather than trusting the producer.
- **DLQ as an attack surface** — a dead-letter queue that retries indefinitely can be used to keep a malformed or malicious event "alive" in the system, repeatedly probing the indexer; cap retries and alert on DLQ growth rather than retrying forever.
- **No review window for compliance-sensitive content** — regulated content (medical, financial) that normally goes through a review step before publication may bypass that step if the streaming pipeline treats every CDC event as automatically indexable. Add an explicit `approved` flag check in the stream processor for sources where review is a compliance requirement, rather than assuming CDC-captured writes are pre-approved.

The general lesson: streaming RAG's speed advantage is also its main new risk — anything that relied on a human noticing a batch diff before it went live no longer has that checkpoint, and needs an inline, automated equivalent.

</details>

---

## Q18. Design a streaming RAG system for a live financial news / market-data assistant with a 30-second freshness SLO. `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Requirements:** breaking news and price-sensitive filings must be queryable within 30 seconds (p95) of publication; the system must survive traffic bursts around market-moving events without freshness collapsing; malicious or erroneous feed data must not corrupt the index.

```
Sources: news wire APIs (webhook push) + regulatory filing feed (polling, 5s interval)
   │
   ▼
Kafka topic "market-document-events", partitioned by ticker symbol
   (preserves per-symbol ordering while parallelizing across symbols)
   │
   ▼
Stream processor (N=8 consumer replicas, autoscaled on consumer lag metric)
   ├─ Source-trust filter: only wire-service and filed-with-regulator sources
   │  auto-index; other sources queue for a lightweight moderation check
   ├─ Micro-batch embed (flush_interval=0.2s -- tighter than the Q8 default,
   │  since 30s SLO leaves little room for batching latency)
   └─ Two-phase versioned upsert (Q14) -- never a window with zero active
      vectors for a symbol during a correction/update
   │
   ▼
Vector index (per-symbol metadata for fast filtered retrieval)
   │
   ▼
Freshness dashboard: p50/p95/p99 lag per source, alerting at 20s (before
the 30s SLO is actually breached) with automatic consumer autoscale trigger
```

Key decisions: partitioning by ticker symbol (not a random hash) keeps ordering guarantees meaningful at the level users actually query ("show me the latest on AAPL"); autoscaling on consumer lag rather than fixed replica count absorbs the bursts that are this domain's defining characteristic (an earnings call or Fed announcement can 50x event volume for minutes); and the source-trust filter (Q17) is non-negotiable for a financial assistant, where a single unvetted "document" incorrectly indexed as fact within seconds of publication is a materially different risk than the same mistake surviving until the next nightly batch review.

</details>

---

## Q19. What is the background and origin of the streaming RAG pattern? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Streaming RAG isn't a single research paper's invention — it's the application of two mature, separate engineering disciplines to the RAG indexing problem:

- **Change Data Capture**, formalized in database replication literature well before RAG existed, and popularized for general-purpose use by Debezium (an open-source CDC platform built on Kafka Connect, first released 2016), which made "stream every database mutation without touching application code" practical outside of specialized replication tooling.
- **Stream processing systems** — Kafka itself (originally built at LinkedIn, open-sourced 2011) established the durable, ordered, partition-parallel event log that every architecture in this file depends on; Flink and Spark Structured Streaming later added exactly-once processing semantics and windowed aggregation on top of that log.

Streaming RAG is best understood as: take the batch RAG indexing pipeline everyone already knows, and replace its trigger from "runs on a schedule" to "runs on an event," using off-the-shelf CDC and stream-processing infrastructure that predates RAG by years. There is no dedicated "streaming RAG" paper to cite — the pattern is an application of general data-engineering practice to a specific ingestion problem.

</details>

---

## Q20. What are the limitations of streaming RAG, and how is the pattern likely to evolve? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Current limitations:

- **Operational complexity is a permanent tax, not a one-time cost** — every additional moving part (broker, consumer group, DLQ, ordering logic) is something that can fail independently and needs its own monitoring, on top of whatever the vector DB and embedding service already require.
- **No free consistency guarantee across chunks** — if a document is chunked into multiple vectors and an update event is only partially processed (some new chunks written, others still pending), queries can see an internally inconsistent document for a short window even with the two-phase swap (Q14) — the swap makes it atomic per-document, not necessarily atomic across documents that reference each other.
- **Freshness SLOs are easy to define and hard to guarantee under adversarial load** — the decision-gate approach (Q12) only validates against traffic patterns you thought to simulate; a genuinely novel burst shape can still degrade freshness in ways the gate didn't anticipate.
- **Vector DBs were largely designed for batch-style bulk loads first** — incremental single-vector upserts at high frequency are a newer usage pattern for some vector databases, and can carry higher per-operation overhead than a bulk load of the same total volume.

Likely evolution: vector databases are increasingly building native CDC-like change feeds and incremental-index data structures purpose-built for high-frequency upserts (rather than treating streaming as bulk-load-called-repeatedly), which should reduce the gap between batch and streaming write efficiency. Expect tighter integration between CDC platforms and vector databases directly (skipping the general-purpose stream-processor middle layer for the common case), and unified batch+streaming index designs that let a single system serve both a nightly bulk reindex and continuous incremental updates without maintaining two separate code paths.

</details>

---

## Real-World Applications

| Application | Domain | Why Streaming RAG Fits |
|---|---|---|
| Breaking news / market data assistant | Finance / Media | Staleness of even a few minutes materially changes the correctness of an answer |
| Live incident-response documentation | SaaS / Ops | Runbooks and status pages change during an active incident; batch lag would surface outdated guidance |
| E-commerce inventory / pricing Q&A | Retail | Stock levels and prices change continuously; stale answers directly cause customer-facing errors |
| Customer support ticket knowledge base | Support | Newly resolved tickets should be searchable by other agents within seconds, not after a nightly batch |
| Regulatory filings search | Legal / Compliance | New filings must be discoverable immediately for compliance monitoring use cases |
