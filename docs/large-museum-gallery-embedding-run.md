# Large museum and gallery embedding run

## Outcome

Produce a new immutable SigLIP 2 bundle from approved museum and gallery image
datasets without changing the currently deployed Met + NGA bundle in place.
Build, validate, and retain each institution as a removable source bundle, then
merge only completed bundles that share the same model, processor, dtype,
normalization, and date contracts.

The first release wave should add the public-domain CMA, AIC, and SMK datasets
to the existing Met + NGA bundle. Gallery, auction, MoMA-image, and other
contemporary sources belong in a second wave and remain blocked until their
image-level agreements explicitly permit embedding and the product's evidence
display.

## Scope and non-goals

In scope:

- official museum bulk files, saved official API responses, and official IIIF
  derivatives;
- contracted gallery, artist-estate, foundation, fair, or auction-house feeds;
- streamed image downloads for embedding, with no artwork pixels retained in
  the final bundle;
- immutable source snapshots, per-image response hashes, source-specific
  rights gates, content-hash reconciliation, and exact bundle merging;
- a quality and capacity pilot followed by a source-by-source full run.

Out of scope:

- crawling collection, gallery, auction, or search-result HTML;
- treating a metadata license, public URL, download button, or image-available
  flag as image permission;
- embedding rows with missing, ambiguous, expired, or unreviewed image rights;
- changing model revision, processor policy, date rules, or dtype during a run;
- mutating or deleting the current production bundle;
- deployment. Promotion is a separate, explicitly authorized operation after
  the candidate passes every gate below.

## Source lanes

| Lane | Sources | Admission basis | Release target |
| --- | --- | --- | --- |
| Existing production | Met + NGA | Existing completed, reconciled bundles | Reuse without re-embedding |
| Museum wave 1 | CMA, AIC, SMK | Implemented official-source adapters and exact public-domain gates | Public candidate bundle |
| Museum backlog | MoMA, Rijksmuseum, others | New source-specific adapter plus image-level rights review | Future public bundle |
| Partner wave 2 | Galleries, artist estates, foundations, fairs, auction houses | Contracted bulk/API/SFTP feed and explicit machine-use grant | Separate partner bundle first |

Do not include MoMA collection-page image links in this run. MoMA records can
remain metadata-only until an image-level permission basis and trusted image
delivery policy have been reviewed and encoded in an adapter.

For a gallery or auction feed, the agreement must separately cover download,
temporary caching, image transformation, embedding generation and storage,
similarity search, evidence-image display, attribution, cloud processing,
retention, revocation, and deletion. Auction lot-display rights should not be
assumed to include those uses.

## Frozen run contract

Before the first production-scale command, record one run ID and freeze:

- every source URL, source kind, snapshot checksum or Git commit, retrieval
  timestamp, and adapter version;
- the reviewed image-rights statement or partner agreement version for every
  source;
- `google/siglip2-base-patch16-224` at revision
  `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`;
- the current production runtime contract: NumPy 2.4.2, Pillow 12.1.1,
  PyTorch 2.12.0, and Transformers 4.57.1;
- float32, L2 normalization, the model's existing processor contract, and the
  repository's existing date rules;
- batch size, device, downloader concurrency, timeout, retry count, image host,
  and request delay for each source;
- immutable input and completed-artifact directories plus dedicated, durable,
  writable work, checkpoint, QA, and log directories.

Suggested naming:

```text
run:              museum-open-YYYY-MM-v1
source corpus:    <institution>-open-YYYY-MM-v1
source bundle:    <institution>-open-YYYY-MM-siglip2-v1
merged candidate: met-nga-cma-aic-smk-YYYY-MM-siglip2-v1
```

If a partner source is admitted, name it explicitly and keep its final bundle
separate until removal, expiry, and display behavior have been exercised.

## Execution plan

### Phase 0: Baseline and go/no-go record

1. Record the current production bundle ID, manifest SHA-256, row count, model
   revision, and retrieval benchmark results. Keep its artifact directory
   read-only.
2. Run the repository test and build suites from the exact code revision that
   will execute the job.
3. Confirm the pinned model is already in the worker's local Hugging Face
   cache. Do not use `--allow-model-download` during the production run.
4. Set capacity and runtime budgets using the pilot formulas below. Record the
   intended worker hardware and software versions.
5. Assign a human approver for source rights, corpus QA, retrieval QA, and final
   promotion. A single failed hard gate stops only that source bundle.
6. Add structured batch/request telemetry with deterministic tests. The current
   CLI reports committed row progress and final input provenance, but its
   manifest does not record per-request latency, retry count, transferred
   bytes, or download/decode/model stage durations; external process counters
   cannot reconstruct those per-request facts.
7. Add and test a deterministic pilot selector if a genuinely stratified pilot
   is required. The current adapters provide seeded global sampling, not quotas
   by century, medium, object type, or date certainty.
8. Preserve availability proof in the final bundle. The embedder, repacker, and
   merger carry checksummed JSON audit payloads and `*.availability.csv` proofs
   into `source-provenance/`; keep the focused carry-through and checksum tests
   in the release gate.

Repository verification commands:

```bash
python3 -m unittest discover -s pipeline/tests -v
PYTHONPATH=service python3 -m unittest discover -s service/tests -v
npm test
npm run check
npm run build
```

Go only when source ownership, output paths, rollback bundle, and stop
conditions are written down.

### Phase 1: Acquire immutable source snapshots

Use the acquisition procedure in `pipeline/README.md`:

- CMA: clone the official Git/LFS repository, check out a pinned commit, pull
  `data.json`, and record the commit and payload checksum;
- AIC: download the official API data archive, verify its upstream freshness,
  and pin the exact archive SHA-256;
- SMK: save the nightly ZIP under an immutable dated name and pin its SHA-256;
- partner source: accept only the exact contracted export and preserve its
  encrypted/private agreement reference, export checksum, delivery time, and
  row scope.

Never overwrite a snapshot in place. A changed upstream payload is a new source
revision and therefore a new run input.

Gate:

- every input is local, immutable, readable, and checksum-pinned;
- every source has an authoritative URL and retrieval timestamp;
- metadata terms and image terms have been reviewed independently;
- confidential partner agreement text stays outside the public artifact while
  its stable ID, version, and checksum remain in protected provenance.

### Phase 2: Run a 1,024-row functional smoke per source

For CMA, AIC, and SMK, run the implemented `prepare-*-visual` command with
`--sample-size 1024` and preflight enabled. Build and embed each smoke corpus
separately with the production model revision. Do not use `--no-preflight` or
`--allow-unreviewed-images`.

Starting operational settings:

| Source | Image host | Downloader settings |
| --- | --- | --- |
| CMA | `openaccess-cdn.clevelandart.org` | Start at a source-approved conservative value; ramp only in the pilot |
| AIC | `www.artic.edu` | Exactly 1 worker and at least 1 second between request starts |
| SMK | `iip-thumb.smk.dk` | Start at a source-approved conservative value; ramp only in the pilot |

The AIC settings apply to both availability preflight and the actual image
downloads. Keep AIC in its own embedding job so its pacing policy does not
throttle other institutions.

Smoke, pilot, and full output basenames otherwise create separate availability
caches. When the pinned source revision and image policy are unchanged, copy
the prior URL-bound `*.availability.csv` to the next phase's expected basename
before preparation and retain its checksum. The adapter will reuse matching
URLs and preflight only new or retryable entries. Never reuse a cache across a
source or policy revision.

Gate:

- the adapter manifest matches the pinned source bytes;
- `prepared_rows` equals the intended reachable sample after declared
  exclusions;
- every row has a stable source record, positive source-specific permission,
  exact `image_rights_uri`, and trusted HTTPS image host;
- CMA reports and excludes every row quarantined by
  `cma-timeline-date-sanity/v1`; no source dates are clipped or silently
  repaired;
- the corpus build reports zero unreviewed images;
- all prepared smoke rows embed, reconcile, repack, load through
  `ArtifactBundle`, and return plausible evidence in a local search smoke; when
  at least 1,024 eligible and reachable rows exist, the prepared sample must
  contain all 1,024;
- a source-stratified human audit finds no restricted images, placeholders,
  creator-lifespan dates substituted for work dates, or broken attribution.

### Phase 3: Run the 10k-25k quality and capacity pilot

Create separate deterministic pilot corpora per source whose combined total is
10,000-25,000 rows. If Phase 0 delivered the stratified selector, allocate
documented quotas across century, medium, object type, date certainty, and
permission class. Otherwise call these seeded per-source samples and audit
their post-hoc coverage; do not claim they are stratified. Use the same model
and source policies planned for the full run, and merge only after the source
pilots have been embedded independently.

Intentionally interrupt the pilot after a committed batch and verify that the
unchanged command resumes at `state.json.completed` without missing or
duplicating rows. Finish with a sustained soak using the proposed full-run
settings so CDN throttling, thermal throttling, memory growth, and checkpoint
flush behavior appear before the full run.

Measure rather than guess:

- successfully embedded rows per hour by source;
- download, decode, and model time per batch;
- retry and terminal failure counts by HTTP status and host;
- peak GPU/MPS memory, host RAM, checkpoint allocation, output growth, and
  network bytes;
- merge and checksum-validation time;
- local service startup memory, cold-query latency, warm-query latency, and
  concurrent-request behavior on the intended deployment class.

Run at least 50 labeled benchmark queries spanning objects, scenes, styles,
emotions, iconography, historically contingent language, and adversarial/null
queries. Review top, borderline, middle, and random results across sources,
periods, and media. Compare precision at K, nDCG, paraphrase stability,
institution-specific failure modes, and concentration-trace stability against
the current Met + NGA baseline.

Gate:

- freeze the fastest configuration that stays within the declared memory,
  network, and source-pacing budgets;
- no unexplained retrieval regression or source-specific quality collapse;
- no rights, host, checksum, row-order, or normalization failure;
- approve the full-run row count, wall-clock estimate, disk reservation, and
  maintenance window.

Changing any frozen setting after this gate creates a new checkpoint and run
record; it must not resume the old checkpoint.

### Phase 4: Prepare and build the full per-source corpora

Run each adapter with `--sample-size 0` and preflight enabled. Then run
`pipeline build` for each source with explicit, non-default provenance:

```bash
python3 -m pipeline build \
  --input <FULL_PREPARED_CSV> \
  --output <NEW_EMPTY_CORPUS_DIR> \
  --corpus-version <IMMUTABLE_SOURCE_CORPUS_ID> \
  --source-revision <PINNED_COMMIT_OR_SHA256> \
  --source-url <OFFICIAL_SOURCE_URL> \
  --source-kind <SOURCE_SPECIFIC_KIND> \
  --counting-unit physical-object \
  --retrieved-at <ISO_8601_TIMESTAMP> \
  --source-payload <ADAPTER_MANIFEST> \
  --source-payload <AVAILABILITY_STATE> \
  --require-parquet
```

Use each adapter's reviewed metadata-license declaration from
`pipeline/README.md`; do not label SMK's Public Domain Mark as a metadata
license. Do not pass different date bounds for one source, because incompatible
date rules will prevent the final merge.

Gate:

- exact prepared/build counts and every exclusion reason are reviewed;
- `unreviewed_images=0` and all permitted rows have nonempty exact rights URIs;
- artwork IDs are unique, offsets contiguous, source payloads bundled, and all
  emitted files checksummed;
- coverage is summarized by source, century, medium, object type, date
  certainty, and rights status before embedding begins. The built-in
  `coverage.csv` does not include every one of those dimensions, so emit a
  separate checksummed QA report for century and permission-status coverage.

### Phase 5: Embed one institution at a time

Use a dedicated durable checkpoint for each source. A generic invocation is:

```bash
python3 -m pipeline embed \
  --corpus-dir <SOURCE_CORPUS_DIR> \
  --output <NEW_EMPTY_BOOTSTRAP_BUNDLE_DIR> \
  --encoder siglip2 \
  --model google/siglip2-base-patch16-224 \
  --model-revision 75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2 \
  --dtype float32 \
  --batch-size <FROZEN_PILOT_VALUE> \
  --device <FROZEN_PILOT_DEVICE> \
  --download-workers <FROZEN_SOURCE_VALUE> \
  --image-fetch-retries <FROZEN_SOURCE_VALUE> \
  --image-request-timeout <FROZEN_SOURCE_VALUE> \
  --image-host <SOURCE_HOST> \
  --checkpoint-dir <DURABLE_SOURCE_CHECKPOINT_DIR> \
  --no-build-faiss
```

For AIC, additionally require:

```bash
--download-workers 1 \
--image-request-delay-seconds 1 \
--image-host www.artic.edu
```

The CLI prints `embedded=<completed>/<total>` after every committed batch, and
the checkpoint `state.json` records the last committed offset. On interruption:

- if output is absent, rerun the exact same command;
- if output exists and checksum-loads, treat it as atomically published and
  preserve any leftover checkpoint for inspection;
- if output exists but fails validation, quarantine it and investigate before
  starting with a new output path.

Do not edit checkpoint files, mix sources in one checkpoint, change
batch/runtime settings, or point a new corpus at an old checkpoint.

Run independent source jobs in parallel only when they have separate compute,
disk, checkpoints, and network budgets. Otherwise run sequentially. The source
bundle—not an arbitrary row shard—is the supported unit of parallelism.

### Phase 6: Reconcile exact image bytes

For every source:

1. Run `derive-embedded-corpus` against the completed bootstrap bundle and the
   adapter's visual manifest.
2. Rebuild the canonical corpus with the derived CSV and all audit manifests
   as source payloads.
3. If artwork count and ordered IDs are unchanged, use
   `repack-embedded-bundle` to attach the existing vectors to the reconciled
   corpus.
4. If a placeholder or other exclusion changes count or order, do not repack.
   For CMA, AIC, and SMK, derivation currently has no declared placeholder list;
   a newly discovered placeholder therefore requires a reviewed adapter/policy
   update and a new prepare/build/embed cycle. After any necessary second
   embedding, derive its exact inputs, build that derived CSV into a new corpus
   directory, and repack only when exact ordered-ID and image-hash validation
   succeeds.

Gate:

- every final row's `image_sha256`, `visual_cluster_id`, actual response URL,
  decoded dimensions, and image input policy describe the bytes that produced
  its vector;
- matrix shape is `(row_count, 768)`, float32, finite, and L2-normalized;
- all manifests and source provenance pass byte-count and checksum validation;
- the source final bundle is immutable and independently loadable.

### Phase 7: Merge completed source bundles

Reuse the current completed Met + NGA bundle. Merge it with the final CMA, AIC,
and SMK bundles; add a gallery bundle only if it has passed the partner rights
and removal prerequisites.

```bash
python3 -m pipeline merge-embedded-bundles \
  --bundle <CURRENT_MET_NGA_FINAL_BUNDLE> \
  --bundle <CMA_FINAL_BUNDLE> \
  --bundle <AIC_FINAL_BUNDLE> \
  --bundle <SMK_FINAL_BUNDLE> \
  --output <NEW_EMPTY_MERGED_BUNDLE_DIR> \
  --corpus-version met-nga-cma-aic-smk-YYYY-MM-siglip2-v1 \
  --corpus-label "Met, NGA, CMA, Art Institute of Chicago, and SMK open-access image catalogs"
```

The merge must fail if model, processor, dtype, vector dimension,
normalization, date rules, required rights fields for permitted rows, or
artifact ordering are incompatible. It is not, by itself, a release rights
gate: it can preserve `unreviewed` rows. Assert
`counts.unreviewed_images == 0` in every source build manifest before merging
and again in the merged `corpus-build-manifest.json`. Do not weaken merge or
release checks to admit a source.

Gate:

- merged row count equals the sum of source bundle counts;
- the merged counting unit is `catalog-record`, inherited from the existing
  mixed Met + NGA production bundle, while the new source bundles retain their
  honest `physical-object` unit;
- source counts, hosts, policies, nested provenance, date denominators, and
  checksums survive the merge;
- each institution can be filtered and audited independently;
- removing a partner means rebuilding the merge without that source bundle,
  not deleting vector rows in place.

### Phase 8: Candidate QA and release decision

Run all repository suites, checksum-load the merged bundle, and exercise the
actual local service. Repeat the labeled retrieval benchmark on the merged
candidate and review:

- source and period coverage, including the public-domain deficit in modern
  and contemporary art;
- duplicate-image clusters and institution-specific ranking concentration;
- at least 50 benchmark queries, null queries, paraphrases, and deliberate
  anachronisms;
- selected-period evidence, image attribution, record links, gaps, sparse-bin
  suppression, and denominator accuracy;
- cold/warm latency, memory, and concurrency on the intended deployment
  hardware;
- comparison with the current production bundle for unexplained regressions.

Release only if the candidate is auditable, every evidence image is displayable
under its recorded rights, the retrieval/metric review passes, and operational
budgets are met. Public labels must call this a versioned museum/gallery catalog
sample, not a measure of all art or historical prevalence.

### Phase 9: Promotion and rollback, separately authorized

Promotion is not part of the embedding run. After approval:

1. upload the immutable bundle to a new private, versioned artifact prefix;
2. retain the previous prefix and bundle unchanged;
3. update the Hugging Face startup guard, corpus ID/count/composition, source
   label, matrix dimensions, bin count, and artifact inventory for the new
   candidate;
4. stage only the deployment allowlist and review the bucket sync dry run;
5. deploy to a separate staging Space, or define an explicit frontend cutover
   mechanism; the existing procedure updates one Space directly and is not a
   canary by itself;
6. verify authenticated health, exact corpus/model identity, and arbitrary
   visual queries on staging before routing production traffic;
7. roll back by restoring the previous bundle/profile, never by modifying the
   candidate in place.

Follow `deploy/huggingface/README.md` for the current private-bucket and Space
procedure. Deployment, label changes, secrets, bucket writes, commits, and
pushes require separate authorization.

## Capacity formulas

For `N` rows at 768 float32 dimensions:

```text
embedding matrix bytes = N * 768 * 4 = N * 3,072
merged rows            = 199,474 + new approved rows
AIC preflight floor    = (live HEAD starts - 1) * 1 second
AIC embedding floor   = (GET starts including retries - 1) * 1 second
```

Useful scale points:

- 199,474 vectors occupy about 584 MiB for the matrix alone; the current
  checksummed Met + NGA bundle is about 836 MiB including metadata and
  provenance;
- every additional 100,000 vectors add about 293 MiB of matrix storage;
- one million vectors occupy about 2.86 GiB for the matrix alone;
- 100,000 live AIC preflight requests require roughly 27.8 hours of
  request-start spacing, and 100,000 embedding GETs require another 27.8 hours.
  Exact totals depend on availability-cache hits, unreachable candidates,
  retries, fallback requests, transfers, decoding, and inference; preflight
  request count can exceed the final prepared-row count.

Reserve working disk phase by phase. If `E_s = source_rows * 3,072`, an active
source embed without FAISS needs approximately `2 * E_s` for its full-size raw
checkpoint and staging/output matrices, plus metadata and decoded-image working
space. During merge, retained source matrices plus the new merged matrix need
approximately `2 * sum(E_s)`, before retained bootstrap/repacked copies and
metadata. Sum active-source peaks when jobs run concurrently. After reserving
those explicit bytes and all snapshots/model caches, leave at least 25% of the
filesystem free; replace this planning margin with the pilot's approved limit.

The checkpoint matrix is allocated at its full size on creation, so its file
size is not a progress signal. Capacity QA must also bound process RSS and
accelerator memory while the current decoded batch and prefetched next batch
coexist. Reduce batch size or the image pixel cap before the full run if the
pilot cannot preserve the approved headroom.

For CMA and SMK, estimate wall time from pilot throughput:

```text
source hours = eligible source rows / measured successful rows per hour
```

Do not extrapolate AIC from CMA/SMK throughput.

## Monitoring and stop conditions

Monitor per source:

- committed rows versus total, rows/hour, and estimated completion time;
- `state.json.completed`, state modification time, provenance row count,
  timestamped CLI progress, free disk, RAM, accelerator memory, and
  temperature;
- download retries, terminal failures, HTTP status distribution, redirects,
  decode failures, and source-host latency;
- source terms/status pages and partner agreement expiry or revocation events;
- post-run checksum, dimensions, duplicate hashes, and coverage deltas.

Stop and quarantine the affected source when:

- source bytes no longer match the pinned revision;
- rights are ambiguous, metadata-only, expired, revoked, or conflict with a
  copyright notice;
- any production row becomes `unreviewed` or lacks its exact rights URI/credit;
- redirects leave the allowlist, access control would need bypassing, or
  sustained 403/429 responses show the source rejects current pacing;
- restricted placeholders appear or final reconciliation changes row order;
- disk drops below the frozen safety margin, the checkpoint stops advancing,
  or hardware errors corrupt a batch;
- the model/runtime configuration differs from the frozen pilot contract.

Do not silently skip, replace, or relabel a failed source. Finish and validate
the other institution bundles, then either repair the affected adapter under a
new run record or omit that source from the merge.

## Partner-source prerequisite

Before the first gallery or auction embedding job, add a source-specific adapter
and protected rights manifest with:

- licensor, agreement ID/version/checksum, effective and expiry dates;
- row-level scope, embargo, territory, audience, attribution, and display-size
  limits;
- separate grants for download, cache, transform, embed, store, search,
  display, redistribute, and cloud processing;
- revocation contact, takedown process, deletion SLA, and whether derived
  embeddings may survive termination;
- trusted delivery hosts and a versioned image derivative policy.

The current generic `explicitly-permitted` status is an enforcement primitive,
not a complete legal record. The current schema/service also cannot enforce
expiry, territory, audience, display-size, revocation, or internal-only evidence
rules. Partner sources remain blocked until those restrictions have runtime
enforcement and tests, unless the reviewed grant unconditionally covers the
entire current production behavior. A human rights approver must sign the exact
feed and agreement scope before the adapter can set permission. Sources that
permit embeddings but prohibit public evidence images cannot enter the current
public or internal service merely by receiving a different label; they require
an actually enforced restricted serving path.
