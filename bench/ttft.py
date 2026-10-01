# --- Time-to-First-Token Benchmark ---
# Replays the fixed trace (bench/trace.jsonl) against a server, one request at a
# time, and records the time-to-first-token of every request.
#
#   python -m bench.ttft --name ttft_vllm_prefix_cache \
#       --base-url http://localhost:8000/v1 --model Qwen/Qwen3-8B-AWQ
#
# Run it once per serving setup, on a freshly started server each time:
#   prefix caching off  -> every request recomputes the whole prompt (baseline)
#   prefix caching on   -> the shared system prompt + tool schemas, and earlier
#                          steps of the same conversation, are reused
#
# "first step" = the first request of a conversation (shares only the system
# prompt and tool schemas with earlier requests). "later steps" = requests after
# a tool call (share everything up to the previous step).
#
# Outputs:
#   results/<name>/ttft.jsonl      one line per request
#   results/ttft_summary.csv       one row per run
import argparse
import asyncio

from openai import AsyncOpenAI

import eval.offline as offline   # must come before the agent imports

from agent.prompts import get_agent_config
from bench.common import time_first_token, median, percentile

SUMMARY_PATH = offline.RESULTS_DIR / "ttft_summary.csv"


async def replay(client, model, conversations, tools):
    # One throwaway request so server start-up cost isn't charged to the first row.
    await time_first_token(client, model, [{"role": "user", "content": "Hello"}])

    rows = []
    for conversation in conversations:
        for step, end in enumerate(conversation["request_ends"]):
            timing = await time_first_token(client, model, conversation["history"][:end], tools)
            rows.append({"id": conversation["id"], "category": conversation["category"], "step": step, **timing})
    return rows


def summarize(rows, name, model) -> dict:
    all_ttft = [r["ttft_s"] for r in rows]
    first = [r["ttft_s"] for r in rows if r["step"] == 0]
    later = [r["ttft_s"] for r in rows if r["step"] > 0]
    prompt_tokens = sum(r["prompt_tokens"] for r in rows)
    cached_tokens = sum(r["cached_tokens"] for r in rows)

    return {
        "name": name,
        "model": model,
        "requests": len(rows),
        "ttft_median_s": median(all_ttft),
        "ttft_p95_s": percentile(all_ttft, 95),
        "first_step_median_s": median(first),
        "later_steps_median_s": median(later),
        "mean_prompt_tokens": round(prompt_tokens / len(rows)) if rows else 0,
        "cached_token_pct": round(100 * cached_tokens / prompt_tokens, 1) if prompt_tokens else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Replay the trace and measure time-to-first-token.")
    parser.add_argument("--name", required=True, help="Run name; results go to results/<name>/")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Only replay the first N conversations")
    args = parser.parse_args()

    conversations = offline.load_jsonl(offline.TRACE_PATH)
    if args.limit:
        conversations = conversations[:args.limit]
    tools = get_agent_config("v1").tools   # the trace is built with v1

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)
    rows = asyncio.run(replay(client, args.model, conversations, tools))

    run_dir = offline.RESULTS_DIR / args.name
    run_dir.mkdir(exist_ok=True)
    offline.write_jsonl(run_dir / "ttft.jsonl", rows)

    summary = summarize(rows, args.name, args.model)
    offline.append_csv_row(SUMMARY_PATH, summary)

    print(f"\n=== {args.name} ===")
    for key, value in summary.items():
        print(f"  {key:<22} {value}")


if __name__ == "__main__":
    main()
