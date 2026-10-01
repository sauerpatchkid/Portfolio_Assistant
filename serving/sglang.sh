#!/usr/bin/env bash
# Start SGLang with an OpenAI-compatible API on http://localhost:<port>/v1
#
#   bash serving/sglang.sh <model> [port]
#
# SGLang is the alternative serving backend. It reuses KV cache for shared
# prompt prefixes automatically (RadixAttention), so there is no flag for it.
set -euo pipefail

MODEL="${1:?usage: sglang.sh <model> [port]}"
PORT="${2:-8000}"

if [ -n "${SERVING_VENV:-}" ]; then source "$SERVING_VENV/bin/activate"; fi

# --context-length 8192       same context limit as the vLLM runs
# --tool-call-parser qwen25   parse Qwen's <tool_call> blocks into structured tool calls
exec python -m sglang.launch_server \
    --model-path "$MODEL" \
    --port "$PORT" \
    --context-length 8192 \
    --tool-call-parser "${TOOL_PARSER:-qwen25}" \
    ${EXTRA_ARGS:-}
