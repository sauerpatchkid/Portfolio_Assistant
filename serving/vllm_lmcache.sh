#!/usr/bin/env bash
# Start vLLM with LMCache as an external KV-cache layer.
#
#   bash serving/vllm_lmcache.sh <model> [port]
#
# vLLM on its own keeps reusable KV cache only in GPU memory, so it is lost when
# the GPU needs the space. LMCache runs next to vLLM as a separate process and
# keeps copies in two more tiers:
#   L1  CPU RAM      (LMCACHE_L1_GB; 4 GB holds about 27,000 tokens of KV cache for the 8B model)
#   L2  local disk   (LMCACHE_DISK_DIR)
# When a prompt prefix is no longer in GPU memory, vLLM asks LMCache for it and
# loads the KV cache back instead of recomputing it.
set -euo pipefail

MODEL="${1:?usage: vllm_lmcache.sh <model> [port]}"
PORT="${2:-8000}"
LMCACHE_PORT="${LMCACHE_PORT:-5555}"
LMCACHE_L1_GB="${LMCACHE_L1_GB:-4}"
LMCACHE_DISK_DIR="${LMCACHE_DISK_DIR:-/content/lmcache_disk}"

if [ -n "${SERVING_VENV:-}" ]; then source "$SERVING_VENV/bin/activate"; fi

mkdir -p "$LMCACHE_DISK_DIR"

# --chunk-size 256   KV cache is stored in 256-token chunks (a multiple of vLLM's 16-token blocks)
# --l2-adapter fs    plain files on the local filesystem
# --http-port        LMCache's status endpoint defaults to 8080, which Colab already uses
lmcache server \
    --host localhost --port "$LMCACHE_PORT" \
    --http-port "${LMCACHE_HTTP_PORT:-8090}" \
    --l1-size-gb "$LMCACHE_L1_GB" \
    --eviction-policy LRU \
    --chunk-size 256 \
    --l2-adapter "{\"type\": \"fs\", \"base_path\": \"$LMCACHE_DISK_DIR\"}" &
LMCACHE_PID=$!
trap 'kill "$LMCACHE_PID" 2>/dev/null || true' EXIT
sleep 10

# Same vLLM flags as serving/vllm.sh, plus the connector that talks to LMCache.
vllm serve "$MODEL" \
    --port "$PORT" \
    --dtype half \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --enable-prompt-tokens-details \
    --kv-transfer-config "{\"kv_connector\": \"LMCacheMPConnector\", \"kv_role\": \"kv_both\", \"kv_connector_extra_config\": {\"lmcache.mp.host\": \"localhost\", \"lmcache.mp.port\": $LMCACHE_PORT}}" \
    ${EXTRA_ARGS:-}
