---
title: Mnemosyne Visual Search
emoji: 🧠
colorFrom: gray
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Mnemosyne visual-search backend

This is the private, CPU-only Hugging Face Space profile for Mnemosyne's
arbitrary visual search. It runs the exact SigLIP 2 text tower used by the
offline image build, then performs exact inner-product retrieval against the
immutable Met artifact bundle. The Space exposes the existing `/v1/search`,
`/v1/evidence`, `/healthz`, `/livez`, and `/readyz` HTTP contracts on port
7860.

The Space repository is intentionally separate from the Mnemosyne application
repository. Never upload the application repository root: it can contain local
artifacts, environment files, caches, and unrelated deployment configuration.

## Pinned production contract

The startup wrapper rejects a mounted bundle unless its schema, corpus,
model/revision, matrix contract, bin count, and artifact count match this
immutable profile. The service loader then verifies every manifest-declared
byte count and SHA-256 digest before accepting traffic. The complete bundle
facts are:

| Property | Pinned value |
| --- | --- |
| Local bundle | `.local-data/artifacts/met-visual-public-domain-142482-siglip2-f32-v3` |
| Corpus ID/version | `met-visual-public-domain-142482-adaptive-hash-v3` |
| Rows | 142,482 |
| Timeline bins | 1,703 |
| Matrix | `(142482, 768)`, `float32`, L2-normalized |
| Exact index | NumPy flat inner product; no FAISS copy |
| Model | `google/siglip2-base-patch16-224` |
| Model revision | `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` |
| Manifest payloads | 10 files, 622,286,492 declared bytes |
| Whole bundle | 11 files including the manifest, 622,523,574 bytes (about 593.7 MiB) |
| Embedding file SHA-256 | `8bcc7e01781bd4e227d2009703fa25afa13cf1eb7ea454d497588eda4c72a3b5` |

The Docker build downloads only the required model files at that exact
revision into `/home/user/.cache/huggingface`, then transfers ownership to the
UID-1000 runtime user. Runtime inference is offline, and a missing snapshot
causes startup to fail instead of silently downloading another revision.

The private Space ingress is the only bearer-authentication boundary for this
profile. The inner API starts with `--http-auth-mode disabled`; do not add a
`MNEMOSYNE_SERVICE_TOKEN` Space secret. A second bearer scheme cannot be
stacked in the single `Authorization` header that private Space ingress uses.

## Runtime shape

The Docker image uses Python 3.11, the CPU-only PyTorch 2.12.0 wheel, UID 1000,
and one API process bound to `0.0.0.0:7860`. BLAS/OpenMP libraries are limited
to two threads. The read-only bucket is mounted at `/artifacts`; startup copies
only manifest-declared files into `/tmp/mnemosyne-artifacts` and exposes the
completed copy with one atomic rename. The application then verifies every
manifest byte count and checksum while loading it.

The HTTP admission cap is eight in-flight search/evidence requests. That cap
is an overload boundary, not eight-way model computation: the service's global
compute lock still serializes cold text encoding and exact matrix scans. The
higher admission cap lets already-cached queries complete while another cold
query owns the compute lock. Requests beyond the admission cap receive `429`
and `Retry-After: 1`.

## Safe deployment procedure

Run these commands from the Mnemosyne application repository. They create a
private bucket and a private Docker Space; they do not deploy the frontend or
change any domain or DNS setting.

### 1. Use a separate, current Hub CLI environment

The inference image pins Transformers 4.57.1, whose runtime dependency range
uses `huggingface_hub<1`. Bucket and volume commands require the current Hub
CLI. Keep the two environments separate:

```bash
MNEMOSYNE_HF_CLI_VENV="${TMPDIR:-/tmp}/mnemosyne-hf-cli-1.27.0"
python3 -m venv "$MNEMOSYNE_HF_CLI_VENV"
source "$MNEMOSYNE_HF_CLI_VENV/bin/activate"
python -m pip install --upgrade pip "huggingface_hub==1.27.0"
hf version
hf auth login
hf auth whoami
```

Use a write-capable, preferably fine-grained token for deployment. Let
`hf auth login` store it; do not paste a token into this README, pass it as a
command-line argument, print it, or put it in the Space repository.

Set non-secret deployment identifiers explicitly:

```bash
export MNEMOSYNE_HF_NAMESPACE="your-hugging-face-user-or-org"
export MNEMOSYNE_HF_BUCKET_ID="$MNEMOSYNE_HF_NAMESPACE/mnemosyne-artifacts"
export MNEMOSYNE_HF_SPACE_ID="$MNEMOSYNE_HF_NAMESPACE/mnemosyne-visual-search"
export MNEMOSYNE_ARTIFACT_BUNDLE="$PWD/.local-data/artifacts/met-visual-public-domain-142482-siglip2-f32-v3"
```

Before continuing, confirm that `MNEMOSYNE_ARTIFACT_BUNDLE` contains the
manifest whose facts are listed above. Do not substitute an embedding-build
checkpoint directory or an older repack.

### 2. Create and populate the private artifact bucket

```bash
hf buckets create "$MNEMOSYNE_HF_BUCKET_ID" --private --exist-ok
hf buckets info "$MNEMOSYNE_HF_BUCKET_ID" --format json |
  python -c 'import json,sys; item=json.load(sys.stdin); assert item["private"] is True, "bucket is not private"; print("verified private bucket:", item["id"])'
hf buckets sync \
  "$MNEMOSYNE_ARTIFACT_BUNDLE" \
  "hf://buckets/$MNEMOSYNE_HF_BUCKET_ID" \
  --dry-run
```

Review every JSONL action from the dry run. It should plan the 11 bundle files,
including `model-manifest.json`, and no unrelated local data. Only after that
review, run the same sync without the dry-run flag:

```bash
hf buckets sync \
  "$MNEMOSYNE_ARTIFACT_BUNDLE" \
  "hf://buckets/$MNEMOSYNE_HF_BUCKET_ID"
```

Bucket sync is non-deleting by default. Do not add `--delete`; a deployment
must never remove remote objects merely because the local source path was
wrong or incomplete.

### 3. Create the private CPU Basic Space and mount the bucket read-only

```bash
hf repos create "$MNEMOSYNE_HF_SPACE_ID" \
  --type space \
  --sdk docker \
  --private \
  --flavor cpu-basic \
  --exist-ok

hf spaces info "$MNEMOSYNE_HF_SPACE_ID" --expand private,sdk --format json |
  python -c 'import json,sys; item=json.load(sys.stdin); assert item["private"] is True, "Space is not private"; assert item["sdk"] == "docker", "Space is not Docker"; print("verified private Docker Space:", item["id"])'
hf spaces volumes ls "$MNEMOSYNE_HF_SPACE_ID"
```

Privacy flags are ignored when `--exist-ok` finds an existing repository, so
the assertions above are mandatory. Stop before syncing artifacts, mounting a
volume, or uploading code if either assertion fails.

Stop and reconcile anything returned by `volumes ls`. `hf spaces volumes set`
replaces the Space's complete volume list; it does not append one mount. Once
the intended complete list is known, set the private bucket mount explicitly
read-only:

```bash
hf spaces volumes set "$MNEMOSYNE_HF_SPACE_ID" \
  --volume "hf://buckets/$MNEMOSYNE_HF_BUCKET_ID:/artifacts:ro"
hf spaces volumes ls "$MNEMOSYNE_HF_SPACE_ID"
```

If the Space must retain another existing volume, include that volume in the
same `set` command. Do not run the one-volume example over an unexplained
existing configuration.

### 4. Stage only the Space allowlist, then upload it

The staging helper copies only this Space card, Dockerfile, startup wrapper,
`service/pyproject.toml`, and the runtime package's top-level `.py` files. It
does not copy tests, caches, local data, `.env` files, the frontend, or the
unrelated container drafts under `service/`.

```bash
export MNEMOSYNE_SPACE_STAGE="$(mktemp -d)"
bash deploy/huggingface/stage-space.sh "$MNEMOSYNE_SPACE_STAGE"
find "$MNEMOSYNE_SPACE_STAGE" -type f -print | LC_ALL=C sort
```

Review the printed allowlist, then upload that directory—not `.` and not the
application repository root:

```bash
hf upload \
  "$MNEMOSYNE_HF_SPACE_ID" \
  "$MNEMOSYNE_SPACE_STAGE" \
  . \
  --repo-type space \
  --commit-message "Deploy pinned Mnemosyne visual search"

hf spaces wait "$MNEMOSYNE_HF_SPACE_ID" --timeout 30m
hf spaces info "$MNEMOSYNE_HF_SPACE_ID" --expand private,sdk,runtime,subdomain
```

If the build or runtime does not settle at `RUNNING`, inspect
`hf spaces logs "$MNEMOSYNE_HF_SPACE_ID"`. Do not enable online runtime model
downloads to mask a snapshot or revision error.

### 5. Verify private ingress and an arbitrary visual query

Resolve the Space subdomain without exposing any credential:

```bash
export MNEMOSYNE_HF_SPACE_URL="$(
  hf spaces info "$MNEMOSYNE_HF_SPACE_ID" --expand subdomain --format json |
  python -c 'import json,sys; print("https://" + json.load(sys.stdin)["subdomain"] + ".hf.space")'
)"
```

The following check reads the token already stored by `hf auth login`; it does
not print it or place it in the process command line. It verifies readiness,
the exact loaded model/corpus, and a two-phrase arbitrary query:

```bash
python - <<'PY'
import json
import os
from urllib.request import Request, urlopen

from huggingface_hub import get_token

base_url = os.environ["MNEMOSYNE_HF_SPACE_URL"].rstrip("/")
token = get_token()
if not token:
    raise SystemExit("No locally stored Hugging Face token; run `hf auth login`.")

def request_json(path, payload=None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        base_url + path,
        data=body,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            **({"Content-Type": "application/json"} if body is not None else {}),
        },
        method="GET" if body is None else "POST",
    )
    with urlopen(request, timeout=120) as response:
        return json.load(response)

health = request_json("/healthz")
assert health["status"] == "ok", health
assert health["mode"] == "embedding", health
assert health["modelVersion"] == "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2", health

result = request_json(
    "/v1/search",
    {"query": "a horse, a sailing ship", "view": "all", "filters": {}},
)
assert result["schemaVersion"] == "mnemosyne.search.v1", result
assert len(result["queries"]) == 2, result["queries"]
print(
    json.dumps(
        {
            "status": "verified",
            "corpusVersion": health["corpusVersion"],
            "modelVersion": health["modelVersion"],
            "queries": [item["label"] for item in result["queries"]],
        },
        indent=2,
    )
)
PY
```

For the frontend proxy, create a separate fine-grained **read** token scoped to
this private Space and store it only in the server-side hosting secret
`MNEMOSYNE_EMBEDDING_SEARCH_SERVICE_TOKEN`. Configure the backend URL as
`MNEMOSYNE_EMBEDDING_SEARCH_SERVICE_URL`. Never expose the read token through a
`NEXT_PUBLIC_` variable or browser request, and never reuse the deployment
write token as an application runtime secret.

Deployment is complete only after the Space reports `RUNNING`, the authenticated
health check succeeds, and the arbitrary query returns the pinned model and
corpus contract. Creating a repository or uploading files alone is not a
verified deployment.
