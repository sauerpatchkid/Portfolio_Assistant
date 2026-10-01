# --- Tool-Calling Engine ---
import json
from dataclasses import dataclass, field

from agent.prompts import BASE_SYSTEM_PROMPT
from agent.tools import TOOLS, execute_tool_call

ROUND_LIMIT_MESSAGE = (
    "I couldn't finish the request within the tool-call limit. "
    "Please try rephrasing or asking for a narrower task."
)


@dataclass
class AgentResult:
    response: str
    history: list
    tool_calls: list = field(default_factory=list)   # [{"name", "arguments", "result"}]
    rounds: list = field(default_factory=list)       # one entry per model call
    hit_round_limit: bool = False


async def run_agent(
    backend,
    user_message: str,
    system_prompt: str = None,
    tools: list = None,
    conversation_history: list = None,
    max_tool_rounds: int = 5,
    max_tokens: int = 512,
    execute_tool=execute_tool_call,
) -> AgentResult:
    """
    Run the full tool-calling loop for one user message.

    The model either answers, or asks for tools; tool results are appended to
    the history and the model is called again, up to max_tool_rounds times.
    Pass the returned history back in as conversation_history for the next turn.

    execute_tool(name, arguments) -> JSON string runs a tool; the eval harness
    swaps it to keep read-only questions from changing the database.
    """
    if system_prompt is None:
        system_prompt = BASE_SYSTEM_PROMPT
    if tools is None:
        tools = TOOLS

    if conversation_history is None:
        history = [{"role": "system", "content": system_prompt}]
    else:
        history = list(conversation_history)
        # If caller passed history without a system prompt, prepend one.
        if len(history) == 0 or history[0].get("role") != "system":
            history.insert(0, {"role": "system", "content": system_prompt})

    history.append({"role": "user", "content": user_message})

    executed = []
    rounds = []

    for _ in range(max_tool_rounds):
        turn = await backend.chat(history, tools=tools, max_tokens=max_tokens)
        rounds.append({
            "prompt_tokens": turn.prompt_tokens,
            "completion_tokens": turn.completion_tokens,
            "cached_tokens": turn.cached_tokens,
            "latency_s": round(turn.latency_s, 4),
            "num_tool_calls": len(turn.tool_calls),
        })

        if not turn.tool_calls:
            history.append({"role": "assistant", "content": turn.content})
            return AgentResult(turn.content, history, executed, rounds)

        history.append({
            "role": "assistant",
            "content": turn.content,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                }
                for tc in turn.tool_calls
            ],
        })

        for tc in turn.tool_calls:
            tool_result = execute_tool(tc["name"], tc["arguments"])
            executed.append({"name": tc["name"], "arguments": tc["arguments"], "result": tool_result})
            history.append({"role": "tool", "tool_call_id": tc["id"], "content": tool_result})

    history.append({"role": "assistant", "content": ROUND_LIMIT_MESSAGE})
    return AgentResult(ROUND_LIMIT_MESSAGE, history, executed, rounds, hit_round_limit=True)
