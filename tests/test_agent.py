import asyncio
import json

from agent.agent import run_agent, ROUND_LIMIT_MESSAGE
from agent.backends import ChatTurn, OpenAIBackend, parse_tool_calls
from agent.db import get_holding_by_ticker
from agent.tools import TOOLS

try:   # the OpenAI SDK moved from httpx to httpx2 in v3
    import httpx2 as httpx
except ImportError:
    import httpx


class ScriptedBackend:
    """Returns a fixed list of turns, and remembers the messages it was sent."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []

    async def chat(self, messages, tools=None, max_tokens=512):
        self.seen.append(list(messages))
        return self.turns.pop(0)


def tool_turn(name, **arguments):
    return ChatTurn(content="", tool_calls=[{"id": "call_1", "name": name, "arguments": arguments}])


def test_agent_runs_tool_then_answers():
    backend = ScriptedBackend([tool_turn("get_stock_price", ticker="NVDA"), ChatTurn(content="NVDA is $231.35.")])

    result = asyncio.run(run_agent(backend, "What is the price of NVDA?"))

    assert result.response == "NVDA is $231.35."
    assert [m["role"] for m in result.history] == ["system", "user", "assistant", "tool", "assistant"]
    assert result.tool_calls[0]["name"] == "get_stock_price"
    assert json.loads(result.tool_calls[0]["result"])["ticker"] == "NVDA"
    assert len(result.rounds) == 2
    # the second model call sees the tool result
    assert backend.seen[1][-1]["role"] == "tool"


def test_agent_keeps_history_across_turns():
    backend = ScriptedBackend([ChatTurn(content="Proceed?"), ChatTurn(content="Done.")])

    first = asyncio.run(run_agent(backend, "Buy 5 shares of MSFT."))
    second = asyncio.run(run_agent(backend, "Yes.", conversation_history=first.history))

    assert [m["role"] for m in second.history] == ["system", "user", "assistant", "user", "assistant"]


def test_agent_stops_at_round_limit():
    backend = ScriptedBackend([tool_turn("get_user_info") for _ in range(5)])

    result = asyncio.run(run_agent(backend, "Loop forever", max_tool_rounds=5))

    assert result.hit_round_limit
    assert result.response == ROUND_LIMIT_MESSAGE
    assert len(result.tool_calls) == 5


def test_confirmation_gate_needs_a_turn_between_proposal_and_execution():
    buy = dict(action="BUY", ticker="MSFT", shares=5)

    # Turn 1: the model tries to execute twice in a row. Both calls only preview.
    backend = ScriptedBackend([tool_turn("execute_trade", **buy), tool_turn("execute_trade", **buy),
                               ChatTurn(content="Shall I go ahead?")])
    first = asyncio.run(run_agent(backend, "Buy 5 shares of MSFT.", confirm_trades=True))
    assert [json.loads(tc["result"])["status"] for tc in first.tool_calls] == ["CONFIRMATION_REQUIRED"] * 2
    assert get_holding_by_ticker("MSFT")["shares"] == 18

    # Turn 2: a different trade is a new proposal, so it is previewed too.
    backend = ScriptedBackend([tool_turn("execute_trade", action="BUY", ticker="MSFT", shares=6),
                               ChatTurn(content="That is a different order. Confirm?")])
    second = asyncio.run(run_agent(backend, "Yes.", conversation_history=first.history, confirm_trades=True))
    assert json.loads(second.tool_calls[0]["result"])["status"] == "CONFIRMATION_REQUIRED"
    assert get_holding_by_ticker("MSFT")["shares"] == 18

    # Turn 3: the trade proposed in turn 1 (same order, written differently) now executes.
    backend = ScriptedBackend([tool_turn("execute_trade", action="buy", ticker="msft", shares=5.0, order_type="MARKET"),
                               ChatTurn(content="Done.")])
    third = asyncio.run(run_agent(backend, "Buy the 5.", conversation_history=second.history, confirm_trades=True))
    assert json.loads(third.tool_calls[0]["result"])["status"] == "SUCCESS"
    assert get_holding_by_ticker("MSFT")["shares"] == 23


def test_preview_reports_guardrail_errors_without_trading():
    backend = ScriptedBackend([tool_turn("execute_trade", action="BUY", ticker="NVDA", shares=1000),
                               ChatTurn(content="That order is too large.")])
    result = asyncio.run(run_agent(backend, "Buy 1000 shares of NVDA.", confirm_trades=True))
    assert "Order too large" in json.loads(result.tool_calls[0]["result"])["error"]


def test_parse_tool_calls_from_raw_text():
    text = 'Sure.\n<tool_call>\n{"name": "get_stock_price", "arguments": {"ticker": "AAPL"}}\n</tool_call>'
    calls = parse_tool_calls(text)
    assert calls == [{"id": "call_0", "name": "get_stock_price", "arguments": {"ticker": "AAPL"}}]
    assert parse_tool_calls("<tool_call>not json</tool_call>") == []


def make_openai_backend(message, usage=None):
    """An OpenAIBackend wired to a fake HTTP server that returns one fixed message."""
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": usage or {"prompt_tokens": 2000, "completion_tokens": 30, "total_tokens": 2030},
        })

    backend = OpenAIBackend("http://test/v1", "test-model")
    backend.client = backend.client.with_options(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    return backend, captured


def test_openai_backend_reads_structured_tool_calls():
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_abc",
            "type": "function",
            "function": {"name": "get_stock_price", "arguments": '{"ticker": "NVDA"}'},
        }],
    }
    usage = {"prompt_tokens": 2000, "completion_tokens": 30, "total_tokens": 2030,
             "prompt_tokens_details": {"cached_tokens": 1900}}
    backend, captured = make_openai_backend(message, usage)

    turn = asyncio.run(backend.chat([{"role": "user", "content": "price of NVDA?"}], tools=TOOLS))

    assert turn.tool_calls == [{"id": "call_abc", "name": "get_stock_price", "arguments": {"ticker": "NVDA"}}]
    assert (turn.prompt_tokens, turn.completion_tokens, turn.cached_tokens) == (2000, 30, 1900)
    # the request carries the tools and turns thinking off
    assert len(captured["body"]["tools"]) == 13
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["body"]["temperature"] == 0.0


def test_openai_backend_falls_back_to_raw_tool_call_text():
    raw = '<think>hmm</think><tool_call>\n{"name": "get_user_info", "arguments": {}}\n</tool_call>'
    backend, _ = make_openai_backend({"role": "assistant", "content": raw})

    turn = asyncio.run(backend.chat([{"role": "user", "content": "cash?"}], tools=TOOLS))

    assert turn.tool_calls == [{"id": "call_0", "name": "get_user_info", "arguments": {}}]
    assert turn.content == ""
