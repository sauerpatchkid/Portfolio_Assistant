# --- Run the Eval Suite ---
# Runs every question through the agent against one backend and scores it.
#
#   python -m eval.run_eval --name vllm_8b_c16 \
#       --base-url http://localhost:8000/v1 --model Qwen/Qwen3-8B-AWQ --concurrency 16
#
# Read-only questions run concurrently (--concurrency). Trade questions change
# the database, so they run one at a time with a database reset before each.
#
# Outputs:
#   results/<name>/records.jsonl   one line per question (answer, tools, score, timings)
#   results/summary.csv            one row per run (accuracy, wall time, throughput)
import argparse
import asyncio
import json
import time
from collections import defaultdict

import eval.offline as offline   # must come before the agent imports

from agent.agent import run_agent
from agent.backends import OpenAIBackend, HFBackend
from agent.db import reset_database, get_user_info, get_holdings
from agent.prompts import get_prompt_config
from agent.tools import execute_tool_call
from eval.score import score_question

SUMMARY_PATH = offline.RESULTS_DIR / "summary.csv"


def read_state() -> dict:
    return {
        "cash": get_user_info()["cash_balance"],
        "holdings": {h["ticker"]: h["shares"] for h in get_holdings()},
    }


def execute_tool_read_only(name: str, arguments: dict) -> str:
    """
    Tool executor for read-only questions. They run concurrently against one
    database, so a model that wrongly places a trade must not change the data
    the other questions are reading. The attempt is still recorded and scored.
    """
    if name == "execute_trade":
        return json.dumps({"error": "Trading is disabled for this request."})
    return execute_tool_call(name, arguments)


async def run_question(backend, question, system_prompt, tools, max_tokens):
    """Run every turn of one question and score it."""
    start = time.perf_counter()
    history = None
    turns = []
    rounds = []
    error = None
    is_trade = question["state"] is not None

    try:
        for user_message in question["turns"]:
            result = await run_agent(
                backend, user_message,
                system_prompt=system_prompt, tools=tools,
                conversation_history=history, max_tokens=max_tokens,
                execute_tool=execute_tool_call if is_trade else execute_tool_read_only,
            )
            history = result.history
            turns.append({
                "response": result.response,
                "tools": [tc["name"] for tc in result.tool_calls],
                "tool_calls": [{"name": tc["name"], "arguments": tc["arguments"]} for tc in result.tool_calls],
            })
            rounds.extend(result.rounds)
    except Exception as e:   # a failed request counts as a wrong answer, not a crashed run
        error = f"{type(e).__name__}: {e}"
        while len(turns) < len(question["turns"]):
            turns.append({"response": "", "tools": [], "tool_calls": []})

    final_state = read_state() if is_trade else None
    score = score_question(question, turns, final_state)

    return {
        "id": question["id"],
        "category": question["category"],
        "split": question["split"],
        "min_tool_calls": question["min_tool_calls"],
        "turns": turns,
        "rounds": rounds,
        "latency_s": round(time.perf_counter() - start, 3),
        "error": error,
        **score,
    }


async def run_suite(backend, questions, system_prompt, tools, concurrency, max_tokens):
    read_only = [q for q in questions if q["state"] is None]
    trades = [q for q in questions if q["state"] is not None]

    reset_database()
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(question):
        async with semaphore:
            return await run_question(backend, question, system_prompt, tools, max_tokens)

    start = time.perf_counter()
    records = list(await asyncio.gather(*(bounded(q) for q in read_only)))
    read_only_wall = time.perf_counter() - start

    start = time.perf_counter()
    for question in trades:
        reset_database()
        records.append(await run_question(backend, question, system_prompt, tools, max_tokens))
    trade_wall = time.perf_counter() - start
    reset_database()

    return records, read_only_wall, trade_wall


def summarize(records, args, read_only_wall, trade_wall) -> dict:
    total = len(records)
    correct = sum(r["correct"] for r in records)
    multi = [r for r in records if r["min_tool_calls"] >= 2 or r["category"] == "trade"]
    all_rounds = [rnd for r in records for rnd in r["rounds"]]
    completion_tokens = sum(rnd["completion_tokens"] for rnd in all_rounds)
    wall = read_only_wall + trade_wall

    return {
        "name": args.name,
        "backend": args.backend,
        "model": args.model,
        "prompt": args.prompt,
        "split": args.split,
        "concurrency": args.concurrency,
        "questions": total,
        "correct": correct,
        "accuracy_pct": round(100 * correct / total, 1) if total else 0.0,
        "multi_step_questions": len(multi),
        "multi_step_correct": sum(r["correct"] for r in multi),
        "errors": sum(1 for r in records if r["error"]),
        "wall_s": round(wall, 1),
        "read_only_wall_s": round(read_only_wall, 1),
        "trade_wall_s": round(trade_wall, 1),
        "model_calls": len(all_rounds),
        "prompt_tokens": sum(rnd["prompt_tokens"] for rnd in all_rounds),
        "cached_tokens": sum(rnd["cached_tokens"] for rnd in all_rounds),
        "completion_tokens": completion_tokens,
        "completion_tokens_per_s": round(completion_tokens / wall, 1) if wall else 0.0,
    }


def print_report(records, summary):
    by_category = defaultdict(list)
    for r in records:
        by_category[r["category"]].append(r["correct"])

    print(f"\n=== {summary['name']} ===")
    for category, results in by_category.items():
        print(f"  {category:<18} {sum(results):>3}/{len(results)}")
    print(f"  {'TOTAL':<18} {summary['correct']:>3}/{summary['questions']}  ({summary['accuracy_pct']}%)")
    print(f"  wall time: {summary['wall_s']}s  "
          f"(read-only {summary['read_only_wall_s']}s at concurrency {summary['concurrency']}, "
          f"trades {summary['trade_wall_s']}s sequential)")
    print(f"  throughput: {summary['completion_tokens_per_s']} completion tokens/s")
    if summary["errors"]:
        print(f"  WARNING: {summary['errors']} questions hit a request error (see records.jsonl)")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the eval suite against one backend.")
    parser.add_argument("--name", required=True, help="Run name; results go to results/<name>/")
    parser.add_argument("--backend", choices=["openai", "hf"], default="openai")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--quantized", action="store_true", help="hf backend: load in 4-bit")
    parser.add_argument("--prompt", default="v1", help="Prompt version from agent/prompts.py")
    parser.add_argument("--split", choices=["all", "dev", "test"], default="all")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions")
    return parser.parse_args()


def main():
    args = parse_args()

    questions = offline.load_jsonl(offline.QUESTIONS_PATH)
    if args.split != "all":
        questions = [q for q in questions if q["split"] == args.split]
    if args.limit:
        questions = questions[:args.limit]

    system_prompt, tools = get_prompt_config(args.prompt)

    if args.backend == "hf":
        # In-process generation handles one request at a time.
        args.concurrency = 1
        backend = HFBackend(args.model, quantized=args.quantized)
    else:
        backend = OpenAIBackend(args.base_url, args.model)

    print(f"Running {len(questions)} questions on {args.model} "
          f"({args.backend}, prompt {args.prompt}, concurrency {args.concurrency})...")

    records, read_only_wall, trade_wall = asyncio.run(
        run_suite(backend, questions, system_prompt, tools, args.concurrency, args.max_tokens)
    )

    run_dir = offline.RESULTS_DIR / args.name
    run_dir.mkdir(exist_ok=True)
    offline.write_jsonl(run_dir / "records.jsonl", records)

    summary = summarize(records, args, read_only_wall, trade_wall)
    offline.append_csv_row(SUMMARY_PATH, summary)
    print_report(records, summary)


if __name__ == "__main__":
    main()
