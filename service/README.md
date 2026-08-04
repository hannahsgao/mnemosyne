# Mnemosyne search service

This package serves both Met catalogue-frequency search and embedding-based
visual retrieval. End users do not supply an API key in either mode.

## Met keyword mode

This is the simplest full Met path and requires only NumPy at runtime:

```bash
python3 -m pip install -e ./service
mnemosyne-search \
  --artifacts /path/to/met-corpus-build \
  --met-keyword \
  --host 127.0.0.1 \
  --port 8765
```

For each comma-separated concept the service makes one keyless Met Collection
API search, intersects the returned object IDs with the local frozen corpus,
and aggregates the precomputed date weights. The plotted value is:

```text
metadata_frequency[j,b] = matching_date_weight[j,b] / eligible_date_weight[b]
```

The bounded per-series cache avoids repeating searches for concepts already
seen. The service fetches object details only for the selected evidence cards;
those calls are cached as well. `--met-search-mode broad` is the default;
`title` and `tags` constrain matching to those catalogue fields. This metric is
catalogue metadata frequency, not a claim about what is visible in each image.

## Embedding mode

The embedding path performs text encoding, exact normalized inner-product
retrieval, fixed global top-1% concentration lift, diagnostics, and evidence
selection. The image tower and image downloads remain offline build concerns.

### Artifact contract

Point the service at a self-contained embedding bundle emitted by
`python3 -m pipeline embed`. The loader reads `model-manifest.json` and its
declared files:

```text
corpus.csv
embeddings.npy
date-weights.npz
bin-denominators.csv
model-manifest.json
```

The loader validates shared row/bin dimensions, normalized date weights,
precomputed denominators, corpus/model versions, and embedding dimensions. It
also accepts the checked-in JSON fixture representation under
`service/tests/fixtures`; JSON embeddings and date weights are for small tests,
not production deployments.

FAISS `IndexFlatIP` is preferred when `faiss-cpu` is installed and the
manifest-declared prebuilt index is loaded when present. NumPy/BLAS is the exact
fallback. A filter is applied while scoring the eligible row set; the service
never retrieves a global top-K and filters it afterward. Exact FAISS subset
indices are retained in a bounded in-process cache for repeated corpus views.

### Run locally

Create a virtual environment, then install the base package plus the desired
local inference/index extras:

```bash
python3 -m pip install -e './service[siglip2,faiss]'
mnemosyne-search \
  --artifacts /path/to/model-artifacts \
  --siglip2 \
  --host 127.0.0.1 \
  --port 8765
```

The pinned model revision must normally be provisioned in the service's local
Hugging Face cache. `--allow-model-download` permits a one-time download at
process setup; neither mode calls hosted inference at query time. The manifest's
model id and revision must exactly match the text tower.

The deterministic fixture service needs no model or FAISS installation:

```bash
PYTHONPATH=service python3 -m mnemosyne_search \
  --artifacts service/tests/fixtures \
  --fixture-vectors service/tests/fixtures/query-embeddings.json \
  --prompt-template '{query}' \
  --prompt-version fixture-prompts-v1 \
  --no-faiss
```

## API

Health check:

```text
GET /healthz
```

Search:

```http
POST /v1/search
Content-Type: application/json

{
  "query": "horse, ship, \"still life, fruit\"",
  "selectedQueryId": "q-...",
  "selectedBinKey": "1850-1899",
  "view": "all",
  "filters": {"institution": ["The Met", "AIC"]}
}
```

Only `query` is required. Commas outside double quotes create independent
series; normalized duplicates are removed in first-seen order, and one to five
unique series are accepted. The response is `mnemosyne.search.v1` and contains
ordered queries, shared corpus/model/metric/bin metadata, one trace per query,
low-signal diagnostics, and evidence for the selected series and bin.

Series cache entries are keyed independently by normalized query, corpus and
model versions, prompt version, view, filters, metric version, percentile, and
counting unit. Selection changes do not invalidate them.

## Verify

```bash
PYTHONPATH=service python3 -m unittest discover -s service/tests -v
python3 -m compileall -q service/mnemosyne_search service/tests
```

The tests use only NumPy and Python's standard library. They cover query syntax,
exact and filtered retrieval, lift math, per-series cache reuse, diagnostics,
filters, selected evidence, strict JSON serialization, and the HTTP boundary.
