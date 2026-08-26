# Offline corpus pipeline

This package builds deterministic corpus artifacts from either a local, pinned
ArtiFact clean-split CSV or the Met's official Open Access export. It never
crawls museum pages.

## Build the Met corpus

The Met adapter is the no-embedding launch path. It selects every record with a
usable normalized date inside the configured bounds; public-domain and image
availability fields remain metadata rather than eligibility gates. It preserves
the original `MetObjects.csv` with a checksum, precomputes sparse date weights
and denominators, and builds a self-contained SQLite FTS5 index used by every
query.

```bash
git clone --depth 1 https://github.com/metmuseum/openaccess.git /path/to/met-openaccess
git -C /path/to/met-openaccess rev-parse HEAD

python3 -m pipeline build-met \
  --input /path/to/met-openaccess/MetObjects.csv \
  --output /path/to/artifacts/met-openaccess-v1 \
  --corpus-version met-openaccess-v1 \
  --source-revision COPY_THE_COMMIT_PRINTED_ABOVE \
  --retrieved-at 2026-08-03T00:00:00Z \
  --min-year -15000 \
  --max-year 2029
```

The output includes `met-search.sqlite3`, whose Porter/Unicode FTS index covers
title, tags, artist, culture, medium, object type, classification, period,
dynasty, geography, and department. The default build makes no network calls
beyond obtaining the CSV separately and downloads neither images nor
embeddings. Pass a saved Met `hasImages` search response with `--image-ids` to
mark image availability, or opt into a one-time snapshot with
`--fetch-image-ids`. Image availability does not change the chart denominator.

At runtime, `mnemosyne-search --met-keyword` opens the SQLite artifact read-only,
so search and aggregation have no Met API dependency. The service uses the
keyless object endpoint only for the small set of selected evidence cards whose
image URLs are not stored in the corpus. `--met-offline-evidence` disables that
optional lookup.

## Build an ArtiFact proof corpus

From the repository root:

```bash
python3 -m pip install -r pipeline/requirements.txt
python3 -m pipeline build \
  --input /path/to/ArtiFact_clean.csv \
  --output /path/to/artifacts/artifact-clean-a8f9b9d \
  --corpus-version artifact-clean-a8f9b9d \
  --source-revision a8f9b9d4ce64d237e997ee39b25aa9768b4dc5ff \
  --retrieved-at 2026-08-03T00:00:00Z \
  --bin-size 10
```

The output directory must be absent or empty. This prevents an immutable
version from being silently modified. `SOURCE_DATE_EPOCH` may be used instead
of `--retrieved-at` in automated builds.

The build always emits CSV tables, a SciPy CSR `date-weights.npz`, the original
source CSV under `source-payloads/`, and a checksum manifest. When PyArrow is
installed it additionally emits the architecture's Parquet artifacts:

```text
corpus.csv                         # portable fallback
corpus.parquet                     # when PyArrow is installed
source-payloads/ArtiFact_clean.csv
images.manifest.csv
images.manifest.parquet            # when PyArrow is installed
date-weights.npz
bin-denominators.csv
bin-denominators.parquet           # when PyArrow is installed
coverage.csv
coverage.parquet                   # when PyArrow is installed
build-manifest.json
```

Use `--require-parquet` in release jobs so a missing PyArrow dependency fails
the build rather than producing only the portable proof outputs.

## Stream a rights-gated Met visual corpus

The Met visual path does not retain artwork pixels. It intersects an official,
pinned Met Open Access corpus with image URLs from a pinned ArtiFact clean CSV,
requires the official Met `public_domain=true` flag, and writes compact metadata
containing the Met object ID and record URL. Omitting `--image-dir` selects the
streamed path:

```bash
python3 -m pipeline prepare-met-visual \
  --met-corpus /path/to/artifacts/met-openaccess-v1 \
  --artifact-csv /path/to/ArtiFact_clean.csv \
  --output-csv /path/to/work/met-visual.csv \
  --sample-size 0 \
  --source-revision PINNED_ARTIFACT_COMMIT \
  --workers 32

python3 -m pipeline build \
  --input /path/to/work/met-visual.csv \
  --output /path/to/artifacts/met-visual-corpus-v1 \
  --corpus-version met-visual-corpus-v1 \
  --source-revision PINNED_ARTIFACT_COMMIT \
  --retrieved-at 2026-08-07T00:00:00Z \
  --metadata-license https://creativecommons.org/publicdomain/zero/1.0/
```

`--sample-size 0` keeps every eligible and reachable work. Before publishing
the CSV, the command performs a resumable, header-only availability check and
records its state beside the output as `*.availability.csv`; this filters stale
URLs without downloading the images. Pass a positive sample size for a seeded
proof corpus. Supplying `--image-dir` opts into a normalized local image cache
instead.

## Build the National Gallery of Art visual corpus

The NGA path consumes only the three official CSVs from a pinned revision of
[`NationalGalleryOfArt/opendata`](https://github.com/NationalGalleryOfArt/opendata):
`objects.csv`, `published_images.csv`, and `object_associations.csv`. Image
eligibility is gated by the image-level `published_images.openaccess=1` flag and
`viewtype=primary`; restricted images are never downloaded, embedded, cached,
or displayed.

Pin the source checkout first:

```bash
git clone https://github.com/NationalGalleryOfArt/opendata.git /path/to/nga-opendata
git -C /path/to/nga-opendata checkout PINNED_NGA_COMMIT
git -C /path/to/nga-opendata rev-parse HEAD
```

Prepare a 1,024-row preflighted smoke corpus before the full run:

```bash
python3 -m pipeline prepare-nga-visual \
  --objects /path/to/nga-opendata/data/objects.csv \
  --published-images /path/to/nga-opendata/data/published_images.csv \
  --object-associations /path/to/nga-opendata/data/object_associations.csv \
  --output-csv /path/to/work/nga-visual-smoke-1024.csv \
  --source-revision PINNED_NGA_COMMIT \
  --sample-size 1024 \
  --workers 32
```

Use `--sample-size 0` for the complete selection after the smoke passes. The
adapter performs a resumable header-only IIIF preflight unless
`--no-preflight` is explicitly supplied. It also:

- excludes virtual objects and records without trustworthy work dates;
- does not substitute an artist lifespan when `displaydate` is missing or
  explicitly unknown;
- chooses duplicate nominal primary images by numeric sequence and UUID;
- maps `relationship=inseparable` children through
  `object_associations.csv` to a shared physical-object root without deleting
  catalog rows;
- requests a bounded NGA IIIF derivative under the versioned policy
  `nga-iiif-fit-1024-short-side-256/v1`.

The 1,024 px derivative is already substantially smaller than the museum
original. It is intentional even though SigLIP's final processor input is 224
px: a local 512-versus-1,024 retrieval check showed enough vector movement to
affect close ranks, while 1,024-versus-2,048 was effectively stable. Keep the
1,024 policy for production unless a larger retrieval benchmark supports a new
versioned policy.

For official revision
`4a1aef41c56f4c20924ffe40898f9ffce000aabf`, the strict full adapter output is
56,992 catalog rows representing 53,829 physical-object IDs. Its manifest
records the source checksums, rights/date exclusions, IIIF policy, association
counts, preflight state, and output checksum.

Build the bootstrap corpus with the same unbounded date-rule configuration as
the production Met bundle. In particular, do not pass `--min-year` or
`--max-year`; a different date contract will be rejected at merge time.

```bash
python3 -m pipeline build \
  --input /path/to/work/nga-visual.csv \
  --output /path/to/artifacts/nga-visual-bootstrap-corpus \
  --corpus-version nga-openaccess-bootstrap-v1 \
  --source-revision PINNED_NGA_COMMIT \
  --source-url https://github.com/NationalGalleryOfArt/opendata \
  --source-kind nga-open-data-local-csv \
  --counting-unit catalog-record \
  --retrieved-at 2026-08-13T10:00:00Z \
  --metadata-license https://creativecommons.org/publicdomain/zero/1.0/ \
  --source-payload /path/to/work/nga-visual.manifest.json \
  --source-payload /path/to/work/nga-visual.availability.csv

python3 -m pipeline embed \
  --corpus-dir /path/to/artifacts/nga-visual-bootstrap-corpus \
  --output /path/to/artifacts/nga-visual-bootstrap-siglip2 \
  --encoder siglip2 \
  --model google/siglip2-base-patch16-224 \
  --model-revision 75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2 \
  --dtype float32 \
  --batch-size 32 \
  --device auto \
  --download-workers 24 \
  --image-host api.nga.gov \
  --no-build-faiss
```

The first embedding pass records the exact response bytes and actual decoded
dimensions. Reconcile those facts into the canonical corpus, then reuse the
already-computed vectors only after exact artwork-order validation:

```bash
python3 -m pipeline derive-embedded-corpus \
  --bundle /path/to/artifacts/nga-visual-bootstrap-siglip2 \
  --visual-manifest /path/to/work/nga-visual.manifest.json \
  --output /path/to/work/nga-visual-content-hashed.csv

python3 -m pipeline build \
  --input /path/to/work/nga-visual-content-hashed.csv \
  --output /path/to/artifacts/nga-visual-content-hashed-corpus \
  --corpus-version nga-openaccess-content-hashed-v1 \
  --source-revision PINNED_NGA_COMMIT \
  --source-url https://github.com/NationalGalleryOfArt/opendata \
  --source-kind nga-open-data-content-hashed-csv \
  --counting-unit catalog-record \
  --retrieved-at 2026-08-13T10:00:00Z \
  --metadata-license https://creativecommons.org/publicdomain/zero/1.0/ \
  --source-payload /path/to/work/nga-visual-content-hashed.manifest.json \
  --source-payload /path/to/work/nga-visual.manifest.json \
  --source-payload /path/to/work/nga-visual.availability.csv

python3 -m pipeline repack-embedded-bundle \
  --embedding-bundle /path/to/artifacts/nga-visual-bootstrap-siglip2 \
  --corpus-dir /path/to/artifacts/nga-visual-content-hashed-corpus \
  --output /path/to/artifacts/nga-visual-siglip2-final
```

`catalog-record` is the honest timeline unit because the NGA selection retains
separately searchable pages and inseparable child records. Their official
physical-object roots remain available for diagnostics and de-duplication.

### Merge completed Met and NGA bundles

The merge operation does not re-embed the Met. It accepts only completed,
content-hash-reconciled bundles with the same model ID, pinned revision,
processor/runtime contract, float32 dimension, normalization, and date rules.
It concatenates vectors in argument order and rebuilds combined bins,
denominators, coverage, offsets, checksums, and nested source provenance.

```bash
python3 -m pipeline merge-embedded-bundles \
  --bundle /path/to/artifacts/met-visual-public-domain-siglip2-final \
  --bundle /path/to/artifacts/nga-visual-siglip2-final \
  --output /path/to/artifacts/met-nga-siglip2-v1 \
  --corpus-version met-nga-siglip2-v1 \
  --corpus-label "The Met and National Gallery of Art open-access image catalog"
```

A mixed Met/NGA output reports `countingUnit=catalog-record`, preserves the
per-row Met and NGA image-input policies, and lists both allowed image hosts.
With the current 142,482-row Met bundle and the strict 56,992-row NGA snapshot,
the full float32 matrix is 199,474 × 768, about 584 MiB before metadata. Run the
full adapter and embedding only after the smoke bundle loads in
`ArtifactBundle` and returns reasonable evidence through the local service.
The production Hugging Face profile is pinned to this completed 199,474-record
merged bundle. Its deployment guard validates the merged corpus ID, label,
catalog-record counting unit, source composition, matrix dimensions, and
artifact inventory; do not deploy the 1,024-row smoke bundle as a release.

## Date rules

The manifest pins `artifact-bootstrap-dates/v1` and all rule parameters.

- Exact dates put total weight `1` in one bin.
- Closed ranges distribute total weight `1` uniformly across their historical
  years and then sum those weights by bin.
- `circa` expands an exact year by five years on each side by default.
- `before` and `after` use a declared 25-year bounded window by default.
- BCE years are negative; year zero is excluded.
- Unknown dates have an all-zero matrix row and do not enter denominators.

`date-weights.npz` is a standard SciPy CSR matrix with corpus rows in
`embedding_offset` order and columns in `bin_index` order from
`bin-denominators.csv`.

### Corpus artifact contract

- `corpus.csv` is ordered by `embedding_offset` and carries stable artwork,
  physical-object, visual-cluster, evidence, provenance, date, and rights
  fields.
- `bin-denominators.csv` is ordered by `bin_index` and provides `bin_key`,
  `bin_start`, `bin_end`, `bin_label`, `eligible_weight`, plus physical-object
  and visual-cluster counts.
- `date-weights.npz` is CSR matrix `W`; row `i` matches corpus
  `embedding_offset=i` and column `b` matches denominator `bin_index=b`.
- `build-manifest.json` contains the corpus `id`, `version`, `count`, and
  `countingUnit`; ordered `bins`; explicit `files` locations; pinned source and
  date rules; counts; and SHA-256 checksums for every emitted artifact.

## Build image embeddings and an exact index

The production adapter loads a pinned SigLIP 2 model from the local Hugging
Face cache unless `--allow-model-download` is explicitly set:

```bash
python3 -m pip install -r pipeline/requirements-embedding.txt
python3 -m pipeline embed \
  --corpus-dir /path/to/artifacts/met-visual-corpus-v1 \
  --output /path/to/artifacts/met-visual-bootstrap-siglip2 \
  --encoder siglip2 \
  --model google/siglip2-base-patch16-224 \
  --model-revision PINNED_HUGGING_FACE_COMMIT \
  --dtype float32 \
  --batch-size 256 \
  --device auto \
  --download-workers 24
```

The image manifest is processed in canonical `embedding_offset` order. A local
`image_path` is used when present; otherwise each `image_url` is fetched into a
bounded in-memory batch, decoded, embedded, hashed, and released immediately.
For the fixed 224 px SigLIP checkpoint, the builder first requests Met's smaller
`web-large` derivative. If either decoded dimension is below the 224 px input
floor, it re-fetches and uses the original instead. It does not assume every
`web-large` response is large enough. The exact response URL, dimensions, input
policy, and SHA-256 are recorded. No artwork pixels enter the final artifact
bundle.

Remote inputs must use HTTPS and an explicit host allowlist. Met's image host is
allowed by default; repeat `--image-host` to support another trusted image
source. `--image-root` resolves relative local `image_path` entries, so the
encoder remains usable with either streamed URLs or an authorized local corpus.

The default hidden checkpoint directory is updated after every successful
batch. Re-running the same command resumes from the last committed offset and
removes the checkpoint only after the immutable bundle has been published.
`--checkpoint-dir` selects a different resumable work directory. Checkpoints are
bound to the corpus, pinned model revision, processor/runtime configuration, and
image-input policy so an incompatible run cannot silently resume them.

Production builds use the default float32 matrix so every exact backend scores
the same deployed values and NumPy can retain its memory map. `embeddings.npy`
is always emitted and is the canonical exact inner-product index. When
`faiss-cpu` is available the builder also writes `index.faiss`; pass
`--no-build-faiss` to omit that optional copy, which is recommended for the
common macOS PyTorch/FAISS wheel combination. Output vectors are L2-normalized,
and the atomically published bundle includes:

```text
embeddings.npy                     # canonical memory-mapped matrix/index
index.faiss                        # optional exact FAISS IndexFlatIP
corpus.csv                         # copied service metadata
date-weights.npz                   # copied W
bin-denominators.csv               # copied denominator/bin contract
corpus-build-manifest.json
model-manifest.json                # model/index settings and all checksums
embedded-images.manifest.csv       # resolved input hashes in embedding order
```

`embedded-images.manifest.csv` records the exact streamed response hash and
resolved source URL, dimensions, and input policy for reproducibility while
leaving `image_path` blank.

`model-manifest.json` repeats the corpus identity and bins and exposes stable
`files.metadata`, `files.embeddings`, `files.dateWeights`, and
`files.binDenominators` keys for the query service. The deterministic encoder
is deliberately marked fixture-only and provides download-free CI coverage:

```bash
python3 -m pipeline embed \
  --corpus-dir /path/to/proof-corpus \
  --output /tmp/proof-index \
  --encoder deterministic
```

Rights are never inferred from image availability. Images lacking a positive
public-domain flag or `image_use_permitted=true` are marked `unreviewed` in the
image manifest and must not enter an embedding/display job without a separate
rights gate.

## Derive and reconcile the production content-hash corpus

Use an initial complete streamed embedding pass to remove known placeholders
before producing the final vectors. Derive a cleaned canonical CSV from the
actual bytes that initial pass received:

```bash
python3 -m pipeline derive-embedded-corpus \
  --bundle /path/to/artifacts/met-visual-bootstrap-siglip2 \
  --visual-manifest /path/to/work/met-visual.manifest.json \
  --output /path/to/work/met-visual-content-hashed.csv
```

The command joins the bundle metadata and embedded-input manifest by
`artwork_id`, removes the known Met placeholder basenames
`Images-Restricted.jpg` and `image-number-only.jpg`, and assigns
`visual_cluster_id=sha256:<input_sha256>`. Multiple physical-object rows remain
in the corpus, but rows backed by identical image bytes now share a visual
cluster for duplicate-aware diagnostics and counts. The output keeps image
paths blank and preserves the actual source URL, dimensions, policy, rights,
and response hash.

The adjacent `*.manifest.json` binds the result to checksums of the source
corpus, embedded-image manifest, model manifest, and visual preflight manifest;
it also carries the original selection/rights gate and exclusion counts.
Because filtering changes row order, `embedding_offset` is deliberately
cleared. Build a new immutable corpus from the derived CSV, passing both small
JSON audit manifests as additional source payloads, then run `pipeline embed`
against that cleaned corpus to produce the final vector matrix:

```bash
python3 -m pipeline build \
  --input /path/to/work/met-visual-content-hashed.csv \
  --output /path/to/artifacts/met-visual-content-hashed-v2 \
  --corpus-version met-visual-content-hashed-v2 \
  --source-revision PINNED_ARTIFACT_COMMIT \
  --source-payload /path/to/work/met-visual-content-hashed.manifest.json \
  --source-payload /path/to/work/met-visual.manifest.json \
  --retrieved-at 2026-08-07T00:00:00Z \
  --metadata-license https://creativecommons.org/publicdomain/zero/1.0/

python3 -m pipeline embed \
  --corpus-dir /path/to/artifacts/met-visual-content-hashed-v2 \
  --output /path/to/artifacts/met-visual-content-hashed-v2-siglip2 \
  --model-revision PINNED_HUGGING_FACE_COMMIT \
  --dtype float32 \
  --batch-size 256 \
  --device auto \
  --download-workers 24 \
  --no-build-faiss
```

`pipeline build` retains the input CSV plus each repeatable `--source-payload`
under `source-payloads/` and checksums them in the corpus manifest. The embedding
builder carries these JSON files into the final model bundle under
`source-provenance/`, where they are covered by the model artifact checksums.

### Reconcile the final streamed hashes without rerunning SigLIP 2

A museum or CDN image URL is not content-addressed: its response bytes can
change between the initial cleanup pass and the final embedding pass even when
the URL remains stable. The final bundle's `embedded-images.manifest.csv` is
therefore the source of truth for the exact bytes that produced each final
vector. Derive once more against that final embedding bundle:

```bash
python3 -m pipeline derive-embedded-corpus \
  --bundle /path/to/artifacts/met-visual-content-hashed-v2-siglip2 \
  --visual-manifest /path/to/work/met-visual.manifest.json \
  --output /path/to/work/met-visual-final-reconciled.csv

python3 -m pipeline build \
  --input /path/to/work/met-visual-final-reconciled.csv \
  --output /path/to/artifacts/met-visual-final-reconciled-v3 \
  --corpus-version met-visual-final-reconciled-v3 \
  --source-revision PINNED_ARTIFACT_COMMIT \
  --source-payload /path/to/work/met-visual-final-reconciled.manifest.json \
  --source-payload /path/to/work/met-visual.manifest.json \
  --retrieved-at 2026-08-07T00:00:00Z \
  --metadata-license https://creativecommons.org/publicdomain/zero/1.0/

python3 -m pipeline repack-embedded-bundle \
  --embedding-bundle /path/to/artifacts/met-visual-content-hashed-v2-siglip2 \
  --corpus-dir /path/to/artifacts/met-visual-final-reconciled-v3 \
  --output /path/to/artifacts/met-visual-final-service-v3
```

`repack-embedded-bundle` does not download images or run either SigLIP tower. It
copies the unchanged float32 `embeddings.npy` and embedded-input manifest,
combines them with the reconciled corpus/date artifacts, carries the checksummed
JSON source provenance forward, and atomically publishes a new model bundle.
Pass `--copy-faiss` only when the source bundle already contains its declared
exact FAISS index and `faiss-cpu` is available to validate it; otherwise the
repacked bundle uses the canonical exact NumPy index.

Repacking deliberately fails closed. It verifies the byte counts and SHA-256s
declared by both source manifests, a finite L2-normalized two-dimensional
float32 matrix, matching manifest/matrix/date-weight row counts, and identical
ordered artwork IDs with contiguous `embedding_offset` values across the source
corpus, embedded provenance, and rebuilt corpus. For every row, the rebuilt
`image_sha256`, `visual_cluster_id`, and image-manifest hash must exactly match
the final embedded `input_sha256`.

The final reconciliation should preserve the already-clean artwork row set. If
it discovers a newly returned placeholder, removes a row, or otherwise changes
ID order, the repacker refuses to attach the old vectors; rebuild and embed that
changed corpus instead.

## Export the static concept catalog

`config/concepts.json` is the versioned, editable concept dictionary. The
exporter consumes a completed embedding bundle and runs only its manifest-pinned
text tower. It never downloads artwork images or invokes the image encoder.

```bash
python3 -m pip install -e './service[siglip2]'
python3 -m pipeline export-concepts \
  --artifacts /path/to/completed-embedding-bundle \
  --concepts config/concepts.json \
  --output public/data/v1 \
  --batch-size 8 \
  --device auto \
  --resume
```

`python3 -m pipeline.export_concepts` accepts the same arguments. The command:

- verifies manifest byte counts/checksums and reuses a private local
  verification stamp on an unchanged resume;
- records corpus, model revision, prompt policy, metric configuration, runtime,
  and export schemas in a content-addressed release;
- searches combined and diagnostic prompt vectors in bounded exact row tiles;
- writes shared bin arrays once and one compact series per concept;
- writes at most one evidence asset per concept, with artwork metadata
  deduplicated across periods;
- preserves the live service's visible `strongest` evidence ordering exactly;
- checkpoints each concept so completed work is skipped on resume.

The small `public/data/v1/manifest.json` pointer may revalidate. Files beneath
its fingerprinted `release` path are immutable and safe for a year-long CDN
cache. Use `--concept-id` or `--limit` for a parity/smoke run; a partial run is
reported honestly in the release manifest and does not replace the public
pointer. Failed full runs behave the same way: their detailed errors stay in
private run output while the release manifest contains only sanitized failure
codes. Only a complete full-catalog run advances the pointer.

## Verify

```bash
python3 -m unittest discover -s pipeline/tests -v
python3 -m pipeline --help
```
