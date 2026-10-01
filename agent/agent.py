# --- Tool-Calling Engine ---
import json
from dataclasses import dataclass, field

from agent.prompts import BASE_SYSTEM_PROMPT
from agent.tools import TOOLS, PREVIEW_FUNCTIONS, CALL_KEYS, execute_tool_call, preview_tool_call

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


def proposed_in_earlier_turn(history: list, name: str, arguments: dict) -> bool:
    """True if the model made this same tool call before the latest user message."""
    last_user = max(i for i, m in enumerate(history) if m["role"] == "user")
    key = CALL_KEYS[name](arguments)

    for message in history[:last_user]:
        for tool_call in message.get("tool_calls") or []:
            function = tool_call["function"]
            if function["name"] != name:
                continue
            try:
                if CALL_KEYS[name](json.loads(function["arguments"])) == key:
                    return True
            except (ValueError, TypeError):
                continue
    return False


async def run_agent(
    backend,
    user_message: str,
    system_prompt: str = None,
    tools: list = None,
    conversation_history: list = None,
    max_tool_rounds: int = 5,
    max_tokens: int = 512,
    execute_tool=execute_tool_call,
    confirm_trades: bool = False,
) -> AgentResult:
    """
    Run the full tool-calling loop for one user message.

    The model either answers, or asks for tools; tool results are appended to
    the history and the model is called again, up to max_tool_rounds times.
    Pass the returned history back in as conversation_history for the next turn.

    execute_tool(name, arguments) -> JSON string runs a tool; the eval harness
    swaps it to keep read-only questions from changing the database.

    confirm_trades turns on the confirmation gate: a trade is only executed if
    the model proposed the same trade in an EARLIER turn, i.e. the user has had
    a chance to reply. The first call gets a preview instead. This holds no
    matter what the model does, so it does not depend on the prompt being obeyed.
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
            name, arguments = tc["name"], tc["arguments"]
            needs_confirmation = (
                confirm_trades
                and name in PREVIEW_FUNCTIONS
                and not proposed_in_earlier_turn(history, name, arguments)
            )
            if needs_confirmation:
                tool_result = preview_tool_call(name, arguments)
            else:
                tool_result = execute_tool(name, arguments)

            executed.append({"name": name, "arguments": arguments, "result": tool_result,
                             "preview": needs_confirmation})
            history.append({"role": "tool", "tool_call_id": tc["id"], "content": tool_result})

    history.append({"role": "assistant", "content": ROUND_LIMIT_MESSAGE})
    return AgentResult(ROUND_LIMIT_MESSAGE, history, executed, rounds, hit_round_limit=True)
