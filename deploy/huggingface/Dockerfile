FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2 \
    VECLIB_MAXIMUM_THREADS=2

RUN groupadd --gid 1000 user \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash user

WORKDIR /home/user/app

COPY --chown=1000:1000 service ./service

RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.12.0" \
    && python -m pip install --no-cache-dir "./service[siglip2]"

# Provision the immutable text-tower snapshot in the same cache the UID-1000
# runtime reads. Space-level preload_from_hub uses a platform build cache that
# is not guaranteed to be mounted into a Docker Space at runtime.
RUN python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/siglip2-base-patch16-224",
    revision="75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    allow_patterns=[
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    ],
)
PY

RUN chown -R 1000:1000 /home/user/.cache

ENV HF_HOME=/home/user/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY --chown=1000:1000 start.py ./start.py

USER 1000:1000

EXPOSE 7860

CMD ["python", "/home/user/app/start.py"]
