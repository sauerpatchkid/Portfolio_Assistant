import asyncio

import eval.offline as offline
from agent.db import get_user_info
from agent.prompts import get_prompt_config
from agent.backends import ChatTurn
from bench.build_trace import build_conversation
from eval.build_questions import build_all
from eval.run_eval import run_suite

from tests.fakes import OracleBackend, SilentBackend


def run(backend, questions, concurrency):
    system_prompt, tools = get_prompt_config("v1")
    records, _, _ = asyncio.run(run_suite(backend, questions, system_prompt, tools, concurrency, 512))
    return records


def test_committed_questions_match_the_builder():
    """questions.jsonl must be rebuilt whenever the snapshot or templates change."""
    committed = offline.load_jsonl(offline.QUESTIONS_PATH)
    assert committed == build_all()


def test_suite_shape():
    questions = build_all()
    assert len(questions) == 100
    assert len({q["id"] for q in questions}) == 100
    multi_step = [q for q in questions if q["min_tool_calls"] >= 2 or len(q["turns"]) > 1]
    assert len(multi_step) == 84
    assert sum(q["split"] == "dev" for q in questions) == 50


def test_perfect_model_scores_100_under_concurrency():
    questions = build_all()
    records = run(OracleBackend(questions), questions, concurrency=16)
    wrong = [(r["id"], r["missing"], r["error"]) for r in records if not r["correct"]]
    assert wrong == []
    assert get_user_info()["cash_balance"] == 15420.50   # database left clean


def test_useless_model_scores_zero():
    questions = build_all()
    records = run(SilentBackend(), questions, concurrency=8)
    assert not any(r["correct"] for r in records)


def test_trade_attempt_in_read_only_question_is_blocked_and_scored_wrong():
    questions = [q for q in build_all() if q["id"] == "cash_after_buy-01"]

    class TradingBackend:
        def __init__(self):
            self.done = False

        async def chat(self, messages, tools=None, max_tokens=512):
            if not self.done:
                self.done = True
                args = {"action": "BUY", "ticker": "META", "shares": 10}
                return ChatTurn(content="", tool_calls=[{"id": "c1", "name": "execute_trade", "arguments": args}])
            return ChatTurn(content="Bought.")

    records = run(TradingBackend(), questions, concurrency=1)
    assert not records[0]["forbidden_ok"]
    assert get_user_info()["cash_balance"] == 15420.50


def test_trace_requests_grow_one_step_at_a_time():
    questions = build_all()
    system_prompt, _ = get_prompt_config("v1")

    multi = next(q for q in questions if q["id"] == "sell_to_buy-01")
    conversation = build_conversation(multi, system_prompt)
    # 3 oracle tool calls -> 3 "what next?" requests + 1 final-answer request
    assert conversation["request_ends"] == [2, 4, 6, 8]
    assert conversation["history"][conversation["request_ends"][-1] - 1]["role"] == "tool"

    trade = next(q for q in questions if q["id"] == "trade-01")
    conversation = build_conversation(trade, system_prompt)
    assert len(conversation["request_ends"]) == 4
    assert [m["role"] for m in conversation["history"]].count("user") == 2
