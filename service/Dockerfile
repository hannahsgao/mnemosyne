# syntax=docker/dockerfile:1.7
FROM python:3.11-slim-bookworm

ARG TARGETARCH
RUN test -z "${TARGETARCH}" || test "${TARGETARCH}" = "amd64"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/model-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MNEMOSYNE_ARTIFACTS=/artifacts \
    MNEMOSYNE_HTTP_AUTH_MODE=required \
    MNEMOSYNE_MAX_CONCURRENT_SEARCHES=1 \
    MNEMOSYNE_LOG_LEVEL=INFO \
    PORT=8080

WORKDIR /opt/mnemosyne/service

COPY requirements-container.txt ./
RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.12.0 \
    && python -m pip install --no-cache-dir -r requirements-container.txt

COPY pyproject.toml README.md ./
COPY mnemosyne_search ./mnemosyne_search
RUN python -m pip install --no-cache-dir --no-deps . \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin mnemosyne \
    && install -d -o mnemosyne -g mnemosyne /artifacts /model-cache

COPY --chmod=755 container-entrypoint.sh /usr/local/bin/container-entrypoint

USER 10001:10001
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/livez', timeout=3).read()"]

ENTRYPOINT ["container-entrypoint"]
