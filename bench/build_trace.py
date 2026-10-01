# --- Build the Benchmark Trace ---
# Turns the eval questions into fixed conversations by following each question's
# "oracle" (the ideal tool calls) and running the real tools. The result is a
# set of multi-step tool-calling conversations that does not depend on any
# model, so every serving setup is timed on exactly the same prompts.
#
#   python -m bench.build_trace           -> writes bench/trace.jsonl
#
# Each line: {"id", "category", "history", "request_ends"}
# Request k of a conversation is history[:request_ends[k]] - the messages a
# server would receive at that point of the tool-calling loop.
import json

import eval.offline as offline   # must come before the agent imports

from agent.db import reset_database
from agent.prompts import get_prompt_config
from agent.tools import execute_tool_call


def build_conversation(question: dict, system_prompt: str) -> dict:
    history = [{"role": "system", "content": system_prompt}]
    request_ends = []
    call_count = 0

    for turn_index, user_message in enumerate(question["turns"]):
        history.append({"role": "user", "content": user_message})
        oracle_turn = question["oracle"][turn_index]

        for tool_call in oracle_turn["calls"]:
            request_ends.append(len(history))   # the model is asked what to do next

            call_id = f"call_{call_count}"
            call_count += 1
            history.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool_call["name"], "arguments": json.dumps(tool_call["arguments"])},
                }],
            })
            history.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": execute_tool_call(tool_call["name"], tool_call["arguments"]),
            })

        request_ends.append(len(history))       # the model is asked for its answer
        if oracle_turn["reply"] is not None:
            history.append({"role": "assistant", "content": oracle_turn["reply"]})

    return {
        "id": question["id"],
        "category": question["category"],
        "history": history,
        "request_ends": request_ends,
    }


if __name__ == "__main__":
    questions = offline.load_jsonl(offline.QUESTIONS_PATH)
    system_prompt, _ = get_prompt_config("v1")

    conversations = []
    for question in questions:
        reset_database()
        conversations.append(build_conversation(question, system_prompt))
    reset_database()

    offline.write_jsonl(offline.TRACE_PATH, conversations)

    requests = sum(len(c["request_ends"]) for c in conversations)
    print(f"Wrote {len(conversations)} conversations ({requests} model requests) to {offline.TRACE_PATH}")
