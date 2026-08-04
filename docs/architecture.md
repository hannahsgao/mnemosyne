# Technical architecture

## Product contract

A query returns two linked objects:

- one or more aligned time series with an explicit metric, named corpus version, bin size, denominator, and uncertainty;
- ranked artworks that explain each time bin, including their source, date certainty, and rights record.

A search may contain one to five comma-separated concepts, such as `horse, ship, train`. Each concept produces an independent line over the same corpus, bins, denominator rules, and metric version. The initial comparison syntax overlays series only; arithmetic operators can be added later.

The evidence trail is not an optional gallery. It is how a user audits whether a line represents the concept they meant.

The product measures a versioned corpus of digitized museum objects, not all art that existed in a period. Corpus composition and coverage are part of every result. End users never provide a museum or model API key.

## Production data flow

### 1. Acquire a versioned corpus

Do not crawl museum web pages. Start from a published, normalized dataset or official institutional bulk data and build immutable corpus snapshots.

#### Bootstrap corpus

The normalized [ArtiFact dataset](https://huggingface.co/datasets/deem-data/ArtiFact) is useful for the embedding proof because it aligns images and metadata from the Met, Rijksmuseum, and Art Institute of Chicago. Use a pinned revision of `ArtiFact_clean`, not the error-injected benchmark split.

ArtiFact is not automatically the production source of truth:

- the [published project](https://olgaovcharenko.github.io/ArtiFact/) is licensed CC BY-NC-ND 4.0, so commercial use and derived artifacts require explicit review or permission;
- its shared schema does not carry all source-record, rights, and credit fields required by this product;
- some normalization is LLM-assisted and must be validated rather than treated as authoritative.

#### Production corpus

Build source adapters for official bulk releases from the [Met](https://github.com/metmuseum/openaccess), [Art Institute of Chicago](https://api.artic.edu/docs/), and [Rijksmuseum](https://data.rijksmuseum.nl/). This is scheduled ETL from published data, not an HTML crawl. Retain source payloads and produce a named snapshot such as `met-aic-rijks-2026-08-v1` with source URLs, retrieval timestamps, checksums, adapter versions, and row counts.

Store one row per physical museum object and group related records instead of deleting them during deduplication:

```text
artwork_id, physical_object_id, visual_cluster_id,
institution, source_id, source_record_url, source_dataset_version,
title, artist, object_type, medium, culture, department,
date_display, date_start, date_end, date_qualifier, date_parse_method,
metadata_license, image_rights_uri, credit_line, public_domain,
image_url, image_sha256, image_width, image_height, embedding_offset
```

Keep the raw source payload beside the canonical table. A `visual_cluster_id` may join duplicate photographs, editions, or depictions while preserving the distinction between separate physical objects. The default counting unit must be stated in the metric; the UI can later offer “museum objects” and “unique visual clusters” as separate corpus views.

Only include images whose terms permit the intended download, embedding, storage, and display behavior. Public-domain-only images will underrepresent modern and contemporary art, so surface rights and digitization coverage by period rather than interpreting a late-period decline as an art-historical fact.

### 2. Normalize dates and precompute weights

An artwork dated 1880–1890 should not become a single false-precision point. Allocate its mass across overlapping bins, with every artwork's date weights summing to one. This prevents a ten-year range from counting ten times more than an exact date.

Preserve the catalog's display date and qualifier. Apply source-specific, versioned rules for `circa`, `before`, `after`, BCE dates, and open-ended ranges. Unknown dates remain eligible for pure similarity search but are excluded from the timeline denominator.

Precompute:

- a sparse artwork-to-bin weight matrix `W`;
- the eligible corpus denominator `D[b]` for every bin and supported corpus filter;
- alternative date-weight matrices for uncertainty sensitivity, such as uniform and center-weighted interpretations;
- coverage summaries by institution, object type, medium, rights status, and date certainty.

Do not describe these bands as uncertainty about all historical art. They measure sensitivity within the observed corpus.

### 3. Embed and index offline

Use [SigLIP 2 through Hugging Face Transformers](https://huggingface.co/docs/transformers/model_doc/siglip2) as the first image-text model. Benchmark the base fixed-resolution checkpoint against the aspect-ratio-preserving NaFlex checkpoint on a stratified sample before embedding the full corpus.

For each permitted primary image:

1. download from the source manifest and verify the content hash;
2. run the versioned image processor and image tower;
3. L2-normalize the embedding;
4. write the embedding in corpus-row order;
5. record the image hash, model revision, processor settings, dtype, and embedding dimension in the build manifest.

Use float32 as the correctness baseline and test float16 or scalar quantization against it before reducing storage. A model or processor change produces a new index version rather than mutating an existing one.

For the initial corpus, use [FAISS `IndexFlatIP`](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) over normalized vectors as the exact production baseline. It provides optimized cosine-equivalent top-K retrieval without introducing an approximate index or vector database. Keep a contiguous memory-mapped matrix and NumPy/BLAS implementation as a simple reference path for full-score experiments.

The offline build emits an immutable artifact bundle:

```text
corpus.parquet
source-payloads/
images.manifest.parquet
embeddings.f16-or-f32
index.faiss
date-weights.npz
bin-denominators.parquet
coverage.parquet
build-manifest.json
```

At request time, the service loads only the text tower, tokenizer, corpus metadata needed for evidence, date artifacts, and vector index. The image tower never runs in the request path. Export or quantize the text tower for CPU inference only after checking ranking parity with the reference model.

The query service owns these model weights and requires no third-party inference API. Cache each normalized series query independently with its model version, corpus version, filters, and aggregation parameters so a request such as `horse, ship` can reuse an existing `horse` result. Run this as a persistent service so model and index loading do not become serverless cold-start work.

Add HNSW, IVF, product quantization, Qdrant, or pgvector only when measured latency, memory, mutation, or concurrency requirements justify them. Any approximate index must be evaluated against the exact index for both retrieval recall and time-series distortion.

### 4. Retrieve and aggregate with global top-percentile lift

For series query `j` with embedding `q[j]`, normalized artwork embedding `v[i]`, date weight `W[i,b]`, and shared bin denominator `D[b]`:

```text
score[j,i] = q[j] dot v[i]
D[b]       = sum over eligible i of W[i,b]
```

Raw similarity is useful for ranking but is not automatically a probability that a concept is present.

The product metric is **global top-percentile concentration lift**. For the first corpus version, set `p = 1%` over the filtered, dated corpus and independently retrieve `K = ceil(p * N)` exact matches for every series. A future percentile change creates a new metric version rather than silently changing existing traces.

Compute:

```text
hit_mass[j,b] = sum over query j's top-K i of W[i,b]
share[j,b]    = hit_mass[j,b] / D[b]
lift[j,b]     = share[j,b] / p
```

`share[j,b]` is the fraction of the period's eligible corpus represented among that series query's strongest global matches. `lift = 1` means the period contains its expected share of those matches; `lift = 3` means they are three times as concentrated in that period. This is the closest initial analogue to [Ngram's denominator-normalized frequency](https://books.google.com/ngrams/info).

Each line receives its own score threshold and top-K set. Series do not compete for a shared result pool, and the same artwork may contribute to multiple concepts. Because every line uses the same percentile, corpus view, bins, and denominators, their lift values are directly comparable around the shared baseline of one.

The chart uses lift so periods and query lines can be compared around a common baseline of one. For each line, the response also returns `hit_mass`, `share`, `D[b]`, and the raw number of physical objects and visual clusters so every point remains auditable.

This metric was chosen because it:

- is robust to query-to-query shifts in raw similarity scale;
- has a clear contribution rule for evidence artworks;
- normalizes for differing corpus sizes across periods;
- can be computed efficiently from exact top-K retrieval and sparse date weights.

Its limitations must remain visible:

- every query, including nonsense, still has a top percentile;
- the curve may be sparse and is sensitive to the percentile choice;
- it measures concentration within this corpus, not absolute historical prevalence.

Return a low-signal warning when top matches do not separate sufficiently from deterministic control samples or when prompt paraphrases produce unstable top-K sets. Suppress or visually de-emphasize bins below the minimum denominator. During validation, compare the fixed 1% trace against 0.5% and 2% sensitivity traces, but do not expose percentile selection as an initial user control.

Every series records the metric identifier and version, percentile, its global score threshold, corpus version, model version, prompt-template version, filters, and counting unit. Retain raw scores and human judgments so calibrated occurrence probabilities can be investigated later, but they are outside the Research MVP.

### 5. Query and evidence flow

For each request:

1. parse commas outside double quotes, trim terms, reject empty terms, deduplicate normalized duplicates, and enforce the five-series limit;
2. preserve each display term and create a small, versioned prompt ensemble for each series;
3. batch-encode all prompts locally, combine each series' prompt embeddings, and normalize the result;
4. select the shared eligible corpus and batch-retrieve an independent exact top-K set for every series;
5. compute concentration share and lift for every series using the same precomputed date weights and denominators;
6. attach per-series low-signal and prompt-stability diagnostics plus shared corpus coverage and date sensitivity;
7. return all traces and their evidence references together, then cache reusable per-series results.

Double quotes allow a literal comma inside one concept, for example `"still life, fruit", horse`. Query parsing is syntax only: the model receives the unquoted concept text. The first release does not implement Ngram-style addition, subtraction, multiplication, division, wildcards, or corpus operators.

Do not apply corpus filters by retrieving a global top-K and filtering afterward; that changes the effective percentile and can distort the trace. For a filtered query, score the eligible row IDs through the memory-mapped matrix or use a separately built exact index for a named, frequently used corpus view. Precompute matching denominators for every supported view.

For a selected series and period, return several deterministic evidence slices:

- strongest contributors;
- representative contributors near the period's median contributing score;
- borderline works around the global threshold;
- deterministic random contributors;
- for low or empty bins, the period's best available non-contributors and a random denominator sample.

Each card links to the museum record and includes its selected series, physical-object and visual-cluster identity, date range and certainty, institution, raw score, contribution weight, and rights status. The timeline legend identifies every series, supports hiding or revealing lines, and makes the selected line explicit before showing evidence.

## Service boundaries

```text
Official datasets -> corpus build -> Parquet + source payloads
Permitted images  -> embedding build -> matrix + FAISS index
Canonical dates   -> date build -> sparse weights + denominators

Comma-separated queries -> batched local text encoder -> per-series exact retrieval
                        -> shared date aggregation -> multi-line timeline
                        -> selected-series evidence drawer
```

The prototype can keep the `/api/search` route, but the production response contract must move timeline construction to the server. It should return an ordered `queries` array plus one typed series per query, along with shared corpus, model, metric, bins, denominators, uncertainty diagnostics, and selected-series evidence rather than only a flat artwork array.

## Libraries and tools

| Need | Start with | Upgrade when measured |
| --- | --- | --- |
| Source corpus | Pinned `ArtiFact_clean` for the proof; official bulk-source adapters for production | Additional institutional adapters or a separately versioned Europeana corpus |
| Tabular ETL and QA | PyArrow + DuckDB + Parquet | Polars or distributed jobs when profiling justifies them |
| Date weights | SciPy sparse matrices | Custom compact representation only if needed |
| Image-text model | Transformers + SigLIP 2 base | A better checkpoint only after benchmark improvement |
| Request-time text inference | Reference Transformers implementation | ONNX Runtime or quantized service after parity tests |
| Exact vector search | FAISS `IndexFlatIP` | HNSW/IVF/PQ after recall and trace-distortion tests |
| Reference scoring | Memory-mapped NumPy/BLAS | GPU batching for offline experiments |
| Dataset review | FiftyOne | Custom review and annotation queues |
| Bulk permitted images | Institutional IIIF and manifest-driven `img2dataset` | Managed object-storage pipeline |
| Mutable product data | None in the analytical path | Postgres for accounts, saved queries, or annotations |
| Chart | Native SVG in this slice | Observable Plot or D3 for comparisons |
| Deep zoom | OpenSeadragon | Mirador for multi-IIIF scholarship |

DuckDB and Parquet remain the source of truth for immutable analytical artifacts. A vector database is not the corpus and is not required for the initial query path.

## Evaluation before scale

### Corpus gates

Before a full embedding run:

- complete a dataset and image-rights review;
- pin all source and model revisions;
- publish corpus composition by century, institution, medium, object type, rights status, and date certainty;
- audit a stratified sample of normalized dates and visual duplicate clusters;
- verify that every displayed artwork can link back to an authoritative record.

### Retrieval benchmark

Create at least 50 benchmark queries spanning objects, scenes, styles, emotions, iconography, historically contingent language, and adversarial negatives. Label stratified top, middle, borderline, and random results across institutions, periods, and media. Track precision at K, nDCG, prompt-paraphrase stability, institution-specific errors, and false-positive behavior for null queries.

### Concentration-metric validation

Validate the fixed 1% concentration-lift metric against the same embeddings and judgments. Include:

- concepts with known broad temporal anchors and deliberate anachronisms;
- sensitivity to 0.5%, 1%, and 2% percentiles, prompt template, date-weight rule, and counting unit;
- agreement between peaks and period-stratified human judgments;
- stability when one institution or medium is removed;
- behavior in sparse bins and on meaningless queries;
- comparability, legend behavior, and selected-series evidence for two-to-five-query overlays;
- batch latency and cache reuse for repeated terms across multi-series requests;
- latency and memory on the intended deployment hardware.

Ship the metric only if the 1% trace is auditable, reasonably stable around the tested percentile, resistant to null-query artifacts, and aligned with period-stratified human judgments. If it fails these gates, fix retrieval or corpus problems before introducing a more complex aggregation method.

The key launch criterion is not a pretty curve. It is whether people trust both the examples under the curve and the denominator behind it.

## Delivery phases

1. **Interaction slice (this repo):** live museum metadata, timeline, evidence cards, mobile UI.
2. **Corpus and rights proof:** pin `ArtiFact_clean` or approved official subsets, define the canonical schema, audit dates and rights, and publish a corpus manifest.
3. **Embedding proof:** embed a 10k–25k stratified sample, compare SigLIP 2 variants, run exact FAISS retrieval, and establish the benchmark.
4. **Concentration proof:** implement the fixed 1% global concentration-lift metric, independent comma-separated series, selected-series evidence, percentile sensitivity, and low-signal detection, then pass the validation gates.
5. **Research MVP:** build 100k+ approved images with date weighting, multi-line concentration lift, coverage panels, comma-separated comparisons, server-computed traces, and a keyless local query service.
6. **Scaled corpus:** add official-source adapters and visual clustering; introduce ANN or a vector service only after measured need.
