# --- Smoke Check for a Running Server ---
# A two-minute check that a serving setup can carry the project, before spending
# time on full runs:
#
#   python -m scripts.smoke_check --name vllm_8b --model Qwen/Qwen3-8B-AWQ
#
# Checks: plain chat works, thinking is off, the server returns structured tool
# calls, the agent completes one single-step and one multi-step question, and
# the server reports token usage. Writes results/smoke/<name>.json.
import argparse
import asyncio
import json
import time

from openai import AsyncOpenAI

import eval.offline as offline   # must come before the agent imports

from agent.agent import run_agent
from agent.backends import OpenAIBackend
from agent.db import reset_database
from agent.prompts import get_agent_config
from bench.common import time_first_token
from eval.score import score_question

NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


async def check(base_url, model):
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=600)
    backend = OpenAIBackend(base_url, model)
    config = get_agent_config("v1")
    system_prompt, tools = config.system_prompt, config.tools
    questions = {q["id"]: q for q in offline.load_jsonl(offline.QUESTIONS_PATH)}
    report = {"model": model, "base_url": base_url, "checks": {}}

    def record(name, passed, detail):
        report["checks"][name] = {"passed": bool(passed), "detail": detail}
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    # 1. Plain chat, thinking disabled
    response = await client.chat.completions.create(
        model=model, max_tokens=64, temperature=0.0, extra_body=NO_THINKING,
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
    )
    text = response.choices[0].message.content or ""
    record("plain_chat", text.strip() != "", repr(text[:80]))
    record("thinking_off", "<think>" not in text, "no <think> block in the reply")

    # 2. Does the server itself return structured tool calls?
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": "What is the current price of NVDA?"}]
    response = await client.chat.completions.create(
        model=model, max_tokens=256, temperature=0.0, extra_body=NO_THINKING,
        messages=messages, tools=tools,
    )
    message = response.choices[0].message
    structured = [(tc.function.name, tc.function.arguments) for tc in (message.tool_calls or [])]
    record("structured_tool_calls", len(structured) > 0,
           str(structured) if structured else f"none; raw content: {(message.content or '')[:200]!r}")

    usage = response.usage
    details = getattr(usage, "prompt_tokens_details", None)
    record("usage_reported", (getattr(usage, "prompt_tokens", 0) or 0) > 0,
           f"prompt_tokens={getattr(usage, 'prompt_tokens', None)}, "
           f"cached_tokens={getattr(details, 'cached_tokens', None)}")

    # 3. Full agent loop on one single-step and one multi-step question
    for question_id in ["lookup-04", "afford-01"]:
        question = questions[question_id]
        reset_database()
        start = time.perf_counter()
        result = await run_agent(backend, question["turns"][0], system_prompt=system_prompt, tools=tools)
        elapsed = time.perf_counter() - start
        turns = [{"response": result.response, "tools": [tc["name"] for tc in result.tool_calls]}]
        score = score_question(question, turns)
        record(f"agent_{question_id}", score["correct"],
               f"tools={turns[0]['tools']}, {elapsed:.1f}s, missing={score['missing']}, "
               f"answer={result.response[:160]!r}")

    # 4. Time-to-first-token on the same prompt twice (the second can reuse the KV cache)
    first = await time_first_token(client, model, messages, tools)
    second = await time_first_token(client, model, messages, tools)
    record("ttft_measured", first["ttft_s"] > 0,
           f"first={first['ttft_s']}s, repeat={second['ttft_s']}s, "
           f"prompt_tokens={first['prompt_tokens']}, cached_tokens_on_repeat={second['cached_tokens']}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Quick health check of a running model server.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    print(f"=== Smoke check: {args.name} ({args.model}) ===")
    try:
        report = asyncio.run(check(args.base_url, args.model))
    except Exception as e:
        report = {"model": args.model, "base_url": args.base_url, "checks": {},
                  "error": f"{type(e).__name__}: {e}"}
        print(f"  [FAIL] request error: {report['error']}")

    smoke_dir = offline.RESULTS_DIR / "smoke"
    smoke_dir.mkdir(exist_ok=True)
    with open(smoke_dir / f"{args.name}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)

    passed = sum(c["passed"] for c in report["checks"].values())
    print(f"  {passed}/{len(report['checks'])} checks passed")


if __name__ == "__main__":
    main()
