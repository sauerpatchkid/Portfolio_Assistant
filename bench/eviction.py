# --- KV-Cache Eviction Benchmark ---
# Shows what happens to a conversation's cached prefix after the server has been
# busy with other work, which is the case an external KV cache (LMCache) is for.
#
#   python -m bench.eviction --name evict_vllm \
#       --base-url http://localhost:8000/v1 --model Qwen/Qwen3-8B-AWQ
#
# For each target conversation (the longest ones in the trace):
#   cold     first time the server sees it: the whole prompt is computed
#   warm     sent again right away: the prompt's KV cache is still in GPU memory
#   revisit  sent again after many unrelated long prompts ("fillers") have
#            pushed it out of GPU memory
#
# Without LMCache, "revisit" should fall back toward "cold". With LMCache the
# evicted KV cache is reloaded from CPU RAM or disk instead of being recomputed.
#
# Outputs:
#   results/<name>/eviction.jsonl      one line per timed request
#   results/eviction_summary.csv       one row per run
import argparse
import asyncio
import copy
import random
import uuid

from openai import AsyncOpenAI

import eval.offline as offline   # must come before the agent imports

from agent.prompts import get_prompt_config
from bench.common import time_first_token, median

SUMMARY_PATH = offline.RESULTS_DIR / "eviction_summary.csv"

FILLER_WORDS = (
    "market price value share stock bond fund cash rate yield growth income asset risk trade "
    "order limit balance sector index return profit loss cost basis total daily weekly annual "
    "report quarter revenue margin debt equity option strike call put volume open close high low"
).split()


def pick_targets(conversations, count):
    """The longest requests in the trace, each made unique so nothing is cached yet."""
    longest = sorted(conversations, key=lambda c: len(str(c["history"])), reverse=True)[:count]
    targets = []
    for conversation in longest:
        messages = copy.deepcopy(conversation["history"][:conversation["request_ends"][-1]])
        # A unique first line means this prompt shares no prefix with anything
        # the server has seen, so the "cold" measurement is really cold.
        messages[0]["content"] = f"Session {uuid.uuid4().hex}.\n" + messages[0]["content"]
        targets.append({"id": conversation["id"], "messages": messages})
    return targets


def make_filler(rng, words):
    text = " ".join(rng.choice(FILLER_WORDS) for _ in range(words))
    return [{"role": "user", "content": f"Note {uuid.uuid4().hex}: {text}\nReply with OK."}]


async def run(client, model, targets, tools, fillers, filler_words, seed):
    rng = random.Random(seed)
    rows = []
    filler_tokens = 0

    await time_first_token(client, model, [{"role": "user", "content": "Hello"}])   # warm-up

    for target in targets:
        for phase in ["cold", "warm"]:
            timing = await time_first_token(client, model, target["messages"], tools)
            rows.append({"target": target["id"], "phase": phase, **timing})

        for _ in range(fillers):
            timing = await time_first_token(client, model, make_filler(rng, filler_words))
            filler_tokens += timing["prompt_tokens"]

        timing = await time_first_token(client, model, target["messages"], tools)
        rows.append({"target": target["id"], "phase": "revisit", **timing})
        print(f"  {target['id']}: " + "  ".join(
            f"{r['phase']}={r['ttft_s']}s" for r in rows if r["target"] == target["id"]
        ))

    return rows, filler_tokens


def main():
    parser = argparse.ArgumentParser(description="Measure TTFT before and after cache eviction.")
    parser.add_argument("--name", required=True, help="Run name; results go to results/<name>/")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--targets", type=int, default=3)
    parser.add_argument("--fillers", type=int, default=24, help="Unrelated prompts sent between visits")
    parser.add_argument("--filler-words", type=int, default=2500, help="Length of each filler prompt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    conversations = offline.load_jsonl(offline.TRACE_PATH)
    _, tools = get_prompt_config("v1")
    targets = pick_targets(conversations, args.targets)

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)
    rows, filler_tokens = asyncio.run(
        run(client, args.model, targets, tools, args.fillers, args.filler_words, args.seed)
    )

    run_dir = offline.RESULTS_DIR / args.name
    run_dir.mkdir(exist_ok=True)
    offline.write_jsonl(run_dir / "eviction.jsonl", rows)

    def phase_median(phase):
        return median([r["ttft_s"] for r in rows if r["phase"] == phase])

    summary = {
        "name": args.name,
        "model": args.model,
        "targets": len(targets),
        "target_prompt_tokens": rows[0]["prompt_tokens"] if rows else 0,
        "filler_tokens_between_visits": round(filler_tokens / len(targets)) if targets else 0,
        "cold_median_s": phase_median("cold"),
        "warm_median_s": phase_median("warm"),
        "revisit_median_s": phase_median("revisit"),
    }
    offline.append_csv_row(SUMMARY_PATH, summary)

    print(f"\n=== {args.name} ===")
    for key, value in summary.items():
        print(f"  {key:<30} {value}")


if __name__ == "__main__":
    main()
