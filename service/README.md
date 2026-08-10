# Mnemosyne search service

This package serves both Met catalogue-frequency search and embedding-based
visual retrieval. End users do not supply an API key in either mode.

## Met keyword mode

This is the simplest full Met path and requires only NumPy plus Python's
built-in SQLite support at runtime:

```bash
python3 -m pip install -e ./service
mnemosyne-search \
  --artifacts /path/to/met-corpus-build \
  --met-keyword \
  --host 127.0.0.1 \
  --port 8765
```

For each comma-separated concept the service searches the frozen artifact's
SQLite FTS5 index, then aggregates the matching rows' precomputed date weights.
The plotted value is:

```text
metadata_frequency[j,b] = matching_date_weight[j,b] / eligible_date_weight[b]
```

The bounded per-series cache avoids repeating searches for concepts already
seen. Search and aggregation stay local; the service calls the keyless Met
object endpoint only for selected evidence cards whose image URLs are absent
from the artifact. `--met-offline-evidence` disables those lookups (such cards
will show no image). `--met-search-mode broad` is the default; `title` and
`tags` constrain matching to those catalogue fields. This metric is catalogue
metadata frequency, not a claim about what is visible in each image. No user
API key is required.

## Embedding mode

The embedding path performs text encoding, exact normalized inner-product
retrieval, score-qualified visual concentration lift, diagnostics, and
period-specific evidence selection. The image tower and image downloads remain
offline build concerns.

### Query and evidence quality policy

The default prompt policy is versioned `art-concept-fixed64-v2`. The SigLIP 2
text encoder reads `text_config.max_position_embeddings` from the pinned model
configuration and tokenizes every prompt with fixed `max_length` padding and
truncation. It fails setup if the model does not declare a positive limit.
Fixed-length input is important for this text tower because dynamic batch
padding changes its pooling position and can make retrieval depend on which
other prompts happen to share the request.

Each query first performs a bounded exact candidate search. The score-qualified
set is then limited to at most `ceil(0.001 * N)` rows and uses the larger of that
rank cutoff and cosine `0.125` as its threshold. The timeline and visible cards
both come from this one set.

For query `j`, let `Q[j]` be its score-qualified rows and let `p_cap = 0.001`.
The service recomputes every plotted period solely from `Q[j]`:

```text
hit_mass[j,b] = sum over i in Q[j] of W[i,b]
share[j,b]    = hit_mass[j,b] / D[b]
lift[j,b]     = share[j,b] / p_cap
```

Object counts and distinct visual-cluster counts use the same qualified rows.
The fixed cap is also the shared lift baseline. If the cosine floor leaves fewer
matches, the missing mass stays missing instead of inflating every survivor;
this keeps amplitudes comparable between queries.
`series.k` and `series.threshold` describe that qualified set; the broader
diagnostic retrieval tail is reported separately as `candidateK` and
`candidateThreshold`.

Cosine similarity is not a calibrated probability, so a score-qualified match
is not automatically a semantically verified one. Dense scores alone cannot
reliably reject nonsense or unsupported language; that requires a separate
lexical or catalogue-attestation signal. The cutoff here is a conservative
retrieval-quality threshold that controls both chart points and cards, not a
general query-validity classifier.

Only periods receiving positive mass from at least two distinct score-qualified
visual clusters are serialized as series points. Bins below the minimum
denominator (20 by default) are also unreliable. Reliable periods with no
qualified mass are drawn at `0×`; positive periods with only one independent
visual cluster and denominator-unreliable periods remain gaps. The web timeline
uses a shape-preserving cubic curve through the exact decade values, holds one
y-domain while panning or zooming, and keeps synthetic zeroes out of evidence
selection. Zero means no qualifying match was found in this indexed corpus and
period—not that the concept was absent from art history.

There is no period-independent evidence strip. `selectedEvidence` supplies only
the strongest score-qualified contributors for the selected plotted period,
de-duplicated by `visual_cluster_id`. Automatic period selection prefers a
reliable period supported by at least three distinct clusters, then falls back
to any plotted period; the cluster count is a selection preference, not a
corpus-wide abstention gate.

### Artifact contract

Point the service at a self-contained embedding bundle emitted by
`python3 -m pipeline embed`. The loader reads `model-manifest.json` and its
declared files:

```text
corpus.csv
embeddings.npy
index.faiss                        # optional exact FAISS IndexFlatIP
date-weights.npz
bin-denominators.csv
embedded-images.manifest.csv
model-manifest.json
```

`embeddings.npy` is the normalized float32 source of truth for exact retrieval
and is loaded as a memory-mapped matrix. No second raw-matrix or NumPy-index
artifact is required. `index.faiss` may also be present when the offline build
was run without `--no-build-faiss`.

The loader validates shared row/bin dimensions, normalized date weights,
precomputed denominators, corpus/model versions, embedding dimensions, and the
declared artifact byte counts and SHA-256 checksums. It
also accepts the checked-in JSON fixture representation under
`service/tests/fixtures`; JSON embeddings and date weights are for small tests,
not production deployments.

Both backends score the same values from `embeddings.npy`. FAISS `IndexFlatIP`
is preferred when `faiss-cpu` is installed; a manifest-declared prebuilt index
is loaded when present, otherwise it can be built in memory. NumPy/BLAS is the
exact fallback and the automatic SigLIP choice on macOS. A filter is applied
while scoring the eligible row set; the service never retrieves an unfiltered
top-K and filters it afterward. Exact FAISS subset indices are retained in a
bounded in-process cache for repeated corpus views.

### Run locally

Create a virtual environment, then install the base package plus the desired
local inference/index extras:

```bash
python3 -m pip install -e './service[siglip2,faiss]'
mnemosyne-search \
  --artifacts /path/to/model-artifacts \
  --siglip2 \
  --device auto \
  --host 127.0.0.1 \
  --port 8766
```

`--device auto` prefers CUDA, then Apple MPS, then CPU for the request-time
text tower. On macOS, SigLIP mode automatically uses exact NumPy/BLAS retrieval
because common FAISS and PyTorch wheels expose conflicting OpenMP runtimes.
`--force-faiss` opts back in after validating a compatible wheel combination.
The bounded candidate and qualification defaults are exposed as `--percentile`,
`--evidence-percentile`, `--minimum-evidence-score`, and
`--minimum-evidence-clusters` and `--minimum-bin-evidence-clusters`. If prompt templates change, assign a new
`--prompt-version`; all of these values participate in the series cache key.

The web toggle runs this process alongside the keyword process on port 8765.
The Next.js route selects the allowlisted service URL from
`searchMode=keyword|embedding`; the Python processes remain independent and do
not need to load each other's artifact bundle.

```dotenv
MNEMOSYNE_SEARCH_MODE=artifact
MNEMOSYNE_KEYWORD_SEARCH_SERVICE_URL=http://127.0.0.1:8765/v1/search
MNEMOSYNE_EMBEDDING_SEARCH_SERVICE_URL=http://127.0.0.1:8766/v1/search
```

The keyword service owns the SQLite FTS artifact on port 8765; the embedding
service owns the SigLIP text tower and exact vector artifact on port 8766. A
failure in one mode is reported for that mode rather than falling through to
the other retrieval semantics.

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

Local images recorded by the embedding manifest are available at
`GET /v1/images/{artworkId}`. Stream-built bundles normally contain no artwork
pixels, so their evidence cards use the compact corpus's remote image URL and
link through `sourceRecordUrl` to the corresponding Met object. The web app
proxies local artifact images through a same-origin, immutable-cache endpoint;
arbitrary filesystem paths are never accepted from HTTP requests.

Only `query` is required. Commas outside double quotes create independent
series; normalized duplicates are removed in first-seen order, and one to five
unique series are accepted. The response is `mnemosyne.search.v1` and contains
ordered queries, shared corpus/model/metric/bin metadata, one trace per query,
low-signal diagnostics, and period-specific `selectedEvidence` with strongest
cards for the selected series and plotted bin. If no dated match passes the
score threshold, the series has no points and `selectedEvidence` is `null`.

Series cache entries are keyed independently by normalized query, corpus and
model versions, prompt version, view, filters, metric version, candidate and
qualification fractions, evidence score/cluster settings, and counting unit.
Selection changes do not invalidate them.

## Verify

```bash
PYTHONPATH=service python3 -m unittest discover -s service/tests -v
python3 -m compileall -q service/mnemosyne_search service/tests
```

The tests use only NumPy and Python's standard library. They cover query syntax,
exact and filtered retrieval, lift math, per-series cache reuse, diagnostics,
fixed-length text tokenization, filters, score-qualified gapped series,
selected-period strongest evidence, abstention, strict JSON serialization, and
the HTTP boundary.
