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
  --corpus-dir /path/to/artifacts/artifact-clean-a8f9b9d \
  --image-root /path/to/downloaded/artifact-images \
  --output /path/to/artifacts/artifact-clean-a8f9b9d-siglip2 \
  --encoder siglip2 \
  --model google/siglip2-base-patch16-224 \
  --model-revision PINNED_HUGGING_FACE_COMMIT
```

The image manifest is processed in canonical `embedding_offset` order. Image
hashes are verified when present, output vectors are L2-normalized, and the
bundle includes:

```text
embeddings.f32 (or embeddings.f16) # contiguous raw matrix
embeddings.npy                     # portable memory-mapped matrix
index-flat-ip.npz                  # exact NumPy inner-product fallback
index.faiss                        # FAISS IndexFlatIP when faiss-cpu is present
corpus.csv                         # copied service metadata
date-weights.npz                   # copied W
bin-denominators.csv               # copied denominator/bin contract
corpus-build-manifest.json
model-manifest.json                # model/index settings and all checksums
embedded-images.manifest.csv       # resolved input hashes in embedding order
```

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

## Verify

```bash
python3 -m unittest discover -s pipeline/tests -v
python3 -m pipeline --help
```
