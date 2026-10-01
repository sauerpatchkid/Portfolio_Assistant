import json

from agent.db import get_user_info, get_holding_by_ticker
from agent.market import get_stock_price, get_historical_data, snapshot_active
from agent.analysis import get_portfolio_summary, get_sector_allocation
from agent.tools import TOOLS, TOOL_FUNCTIONS, execute_tool_call
from agent.trade import execute_trade


def test_registry_is_consistent():
    assert len(TOOLS) == len(TOOL_FUNCTIONS) == 13
    for tool in TOOLS:
        assert tool["function"]["name"] in TOOL_FUNCTIONS


def test_market_data_comes_from_snapshot():
    assert snapshot_active()
    assert get_stock_price("aapl")["ticker"] == "AAPL"
    assert "error" in get_stock_price("NOT_A_TICKER")
    assert get_historical_data("AAPL", "3mo")["period"] == "3mo"
    assert "error" in get_historical_data("AAPL", "2w")


def test_portfolio_summary_adds_up():
    summary = get_portfolio_summary()
    invested = sum(p["current_value"] for p in summary["positions"])
    assert abs(summary["invested_value"] - invested) < 0.05
    assert abs(summary["portfolio_value"] - (invested + summary["cash_balance"])) < 0.05

    allocation = get_sector_allocation()["sector_allocation"]
    assert abs(sum(s["percentage"] for s in allocation) - 100) < 0.1


def test_buy_updates_cash_and_shares():
    price = get_stock_price("MSFT")["current_price"]
    cash_before = get_user_info()["cash_balance"]

    result = execute_trade("BUY", "MSFT", 5)

    assert result["status"] == "SUCCESS"
    assert abs(get_user_info()["cash_balance"] - (cash_before - 5 * price)) < 0.01
    assert get_holding_by_ticker("MSFT")["shares"] == 23


def test_trade_guardrails_block_and_leave_data_unchanged():
    cash_before = get_user_info()["cash_balance"]

    assert "Order too large" in execute_trade("BUY", "NVDA", 1000)["error"]
    assert "restricted" in execute_trade("BUY", "GME", 10)["error"]
    assert "Cannot sell" in execute_trade("SELL", "TSLA", 50)["error"]
    assert "Insufficient funds" in execute_trade("BUY", "MSFT", 100)["error"]

    assert get_user_info()["cash_balance"] == cash_before


def test_execute_tool_call_reports_errors_as_json():
    assert "Unknown tool" in json.loads(execute_tool_call("no_such_tool", {}))["error"]
    assert "argument mismatch" in json.loads(execute_tool_call("get_stock_price", {"symbol": "AAPL"}))["error"]
    assert "offline" in json.loads(execute_tool_call("web_search", {"query": "fed rate"}))["error"]
