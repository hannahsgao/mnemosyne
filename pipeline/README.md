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

## Verify

```bash
python3 -m unittest discover -s pipeline/tests -v
python3 -m pipeline --help
```
