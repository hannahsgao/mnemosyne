#!/bin/sh
set -eu

# Supplying command-line arguments gives operators the full CLI (including the
# deterministic fixture mode). With no arguments, run the private exact SigLIP
# service using mounted artifact and model-cache directories.
if [ "$#" -eq 0 ]; then
  set -- \
    --artifacts "${MNEMOSYNE_ARTIFACTS:-/artifacts}" \
    --siglip2 \
    --device "${MNEMOSYNE_DEVICE:-cpu}" \
    --host "${MNEMOSYNE_HOST:-0.0.0.0}" \
    --port "${PORT:-8080}"
fi

exec mnemosyne-search "$@"
