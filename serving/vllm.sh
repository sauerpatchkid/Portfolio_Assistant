#!/usr/bin/env bash
# Start vLLM with an OpenAI-compatible API on http://localhost:<port>/v1
#
#   bash serving/vllm.sh <model> [port]
#
# EXTRA_ARGS adds flags without editing this file, e.g. the no-cache baseline:
#   EXTRA_ARGS="--no-enable-prefix-caching" bash serving/vllm.sh Qwen/Qwen3-8B-AWQ
set -euo pipefail

MODEL="${1:?usage: vllm.sh <model> [port]}"
PORT="${2:-8000}"

# The server runs from its own virtualenv so its packages can't clash with the notebook's.
if [ -n "${SERVING_VENV:-}" ]; then source "$SERVING_VENV/bin/activate"; fi

# --dtype half                 the Colab T4 has no bfloat16 support, so use float16
# --max-model-len 8192         system prompt + 13 tool schemas + tool results fit comfortably
# --gpu-memory-utilization     share of GPU memory for weights + KV cache
# --enable-auto-tool-choice    let the model decide when to call a tool ...
# --tool-call-parser hermes    ... and parse Qwen's <tool_call> blocks into structured tool calls
# --enable-prompt-tokens-details   report how many prompt tokens came from the prefix cache
# Prefix caching (KV-cache reuse for shared prompt prefixes) is ON by default.
exec vllm serve "$MODEL" \
    --port "$PORT" \
    --dtype half \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --enable-prompt-tokens-details \
    ${EXTRA_ARGS:-}
