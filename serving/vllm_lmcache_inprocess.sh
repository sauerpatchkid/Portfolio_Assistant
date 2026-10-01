#!/usr/bin/env bash
# Fallback: vLLM with LMCache loaded inside the vLLM process (older integration).
# Use this only if serving/vllm_lmcache.sh does not start with the installed versions.
#
#   bash serving/vllm_lmcache_inprocess.sh <model> [port]
#
# Cache tiers are configured in serving/lmcache.yaml.
set -euo pipefail

MODEL="${1:?usage: vllm_lmcache_inprocess.sh <model> [port]}"
PORT="${2:-8000}"

if [ -n "${SERVING_VENV:-}" ]; then source "$SERVING_VENV/bin/activate"; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG_FILE:-$SCRIPT_DIR/lmcache.yaml}"
mkdir -p /content/lmcache_disk

exec vllm serve "$MODEL" \
    --port "$PORT" \
    --dtype half \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --enable-prompt-tokens-details \
    --kv-transfer-config '{"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}' \
    ${EXTRA_ARGS:-}
