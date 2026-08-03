# Technical architecture

## Product contract

A query returns two linked objects:

- a time series with an explicit metric, corpus, bin size, and uncertainty;
- ranked artworks contributing to each time bin, including their source and rights record.

The breadcrumb is not an optional gallery. It is how a user audits whether a line represents the concept they meant.

## Production data flow

### 1. Ingest

Start with open-access museum sources rather than attempting “all artwork” immediately. A practical first corpus is the normalized ArtiFact dataset (Met, Rijksmuseum, and Art Institute of Chicago), or direct institutional dumps/APIs when stronger provenance and rights control are needed.

Store a canonical row per artwork:

```text
artwork_id, institution, source_id, title, artist, date_start, date_end,
image_url, object_url, rights, medium, culture, embedding_offset
```

Retain the original source payload. Normalize museum-specific fields, deduplicate cross-institution records where possible, and never discard ambiguous dates.

### 2. Date weighting

An artwork dated 1880–1890 should not become a single false-precision point. Spread its weight across eligible bins. Circa dates receive a configurable interval; unknown dates are excluded from the timeline but may still appear in pure similarity search.

### 3. Embed

Use SigLIP 2 through Hugging Face Transformers as the first image–text model. Precompute normalized image embeddings in batches, store them as a memory-mapped float matrix, and keep row offsets in Parquet/DuckDB. A text query is embedded at request time in the same vector space.

For the first 100k–500k works, exact matrix multiplication is simple, cheap, and easier to audit than operating a vector database. Add FAISS, Qdrant, or pgvector only after latency measurements justify approximate search or multi-user scale requires a service.

### 4. Retrieve and aggregate

Compute cosine similarity between the query and all eligible images. Do not use one global raw-similarity cutoff: scores vary by query. The first defensible metric is **relative concentration**—the share of artworks in each period that fall within a query-specific top percentile. Always show the period’s corpus denominator and minimum sample thresholds.

Later, train a small calibration layer using human judgments to estimate a more interpretable probability of concept presence. Keep the uncalibrated trace available for comparison.

### 5. Explain every point

For a selected period, return several evidence slices:

- strongest matches;
- representative matches near the period’s median positive score;
- borderline matches around the threshold;
- deterministic random samples from contributors.

This avoids showing only spectacular cherry-picked examples. Each card links to the museum record and includes date certainty, institution, score, and rights status.

## Service boundaries

```text
Ingestion jobs  -> object storage (images) + Parquet (metadata)
Embedding jobs  -> memory-mapped matrix + model/version manifest
Query API       -> text encoder + exact/ANN retrieval + date aggregation
Web app         -> timeline + comparisons + inspectable artwork drawers
```

The prototype’s `/api/search` endpoint already defines the UI-facing seam. Its current museum-metadata implementation can be replaced by the embedding query service without redesigning the client.

## Libraries and tools

| Need | Start with | Upgrade when needed |
| --- | --- | --- |
| Corpus querying | DuckDB + Parquet | Postgres for mutable product data |
| Embeddings | Transformers + SigLIP 2 | Dedicated GPU batch service |
| Vector scoring | NumPy/PyTorch matrix multiply | FAISS, Qdrant, or pgvector |
| Dataset QA | FiftyOne | Custom review queues |
| Bulk images | Institutional IIIF / `img2dataset` | Managed object-storage pipeline |
| Chart | Native SVG in this slice | Observable Plot or D3 for comparisons |
| Deep zoom | OpenSeadragon | Mirador for multi-IIIF scholarship |

## Evaluation before scale

Create a small benchmark of roughly 50 queries spanning objects, scenes, styles, emotions, iconography, and adversarial negatives. For each query, label a stratified sample of top, middle, and random results. Track precision at K, recall proxies, cross-period stability, and institution-specific failure modes.

The key launch criterion is not a pretty curve. It is whether people trust the examples under the curve.

## Delivery phases

1. **Interaction slice (this repo):** live museum metadata, timeline, evidence cards, mobile UI.
2. **Embedding proof:** 10k–25k public-domain images, exact search, fixed benchmark.
3. **Research MVP:** 100k+ images, date weighting, relative concentration, comparison queries, coverage panel.
4. **Scaled corpus:** additional institutions, deduplication, ANN only if measured latency requires it.
