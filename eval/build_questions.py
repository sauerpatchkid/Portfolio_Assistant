# --- Build the 100-Question Eval Suite ---
# Questions come from templates filled in with tickers. Every expected answer is
# COMPUTED here by calling the same tool functions the agent uses (on the mock
# database and the recorded market snapshot), so there is no hand arithmetic.
#
#   python -m eval.build_questions        -> writes eval/questions.jsonl
#
# Each question records:
#   turns            user messages (trade questions have a second "confirm" turn)
#   expected_tools   acceptable tool sets; the agent must call every tool in one of them
#   forbidden_tools  tools that must not be called
#   numbers          values that must appear in the final answer, each with a tolerance
#   contains         keywords that must appear in the final answer
#   state            (trades) expected cash / share count after the conversation
#   oracle           the ideal tool calls per turn; used to build the benchmark trace
#   min_tool_calls   fewest tool calls that can answer it (2+ means multi-step)
import math
from collections import Counter

import eval.offline as offline   # must come before the agent imports

from agent.db import (
    reset_database, get_user_info, get_holdings, get_holding_by_ticker,
    get_recent_transactions,
)
from agent.market import get_stock_price, get_historical_data, get_option_chain
from agent.analysis import get_portfolio_summary, get_sector_allocation, compare_stocks
from agent.trade import execute_trade

# Tolerances for matching a number in the answer
EXACT = 0.01   # share counts, strikes
MONEY = 0.5    # dollars: rounding to the nearest dollar still passes
PCT = 0.6      # percentages: rounding to a whole percent still passes

NO_TRADE = ["execute_trade"]
CONFIRM_MESSAGE = "Yes, I confirm. Please go ahead."
PERIOD_WORDS = {"1mo": "month", "3mo": "3 months", "6mo": "6 months", "1y": "year"}


def num(label, value, tol):
    return {"label": label, "value": round(float(value), 4), "tol": tol}


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


def question(category, text, expected_tools, oracle_calls, min_tool_calls,
             numbers=(), contains=(), contains_match="all"):
    return {
        "category": category,
        "min_tool_calls": min_tool_calls,
        "turns": [text],
        "expected_tools": expected_tools,
        "forbidden_tools": NO_TRADE,
        "expect_no_tools": False,
        "numbers": list(numbers),
        "contains": list(contains),
        "contains_match": contains_match,
        "state": None,
        "oracle": [{"calls": oracle_calls, "reply": None}],
    }


# ── Ground-truth helpers ────────────────────────────────────────────────────

def cash():
    return get_user_info()["cash_balance"]


def price(ticker):
    return get_stock_price(ticker)["current_price"]


def shares_owned(ticker):
    return get_holding_by_ticker(ticker)["shares"]


def pct_change(ticker, period):
    return get_historical_data(ticker, period)["summary"]["pct_change"]


# ── Single-tool lookups (sanity checks) ─────────────────────────────────────

def build_lookups():
    summary = get_portfolio_summary()
    googl = get_stock_price("GOOGL")
    sectors = get_sector_allocation()["sector_allocation"]
    tsla_chain = get_option_chain("TSLA")
    last_trade = get_recent_transactions()[0]
    all_tickers = [h["ticker"] for h in get_holdings()]

    return [
        question("lookup", "What is my current cash balance?",
                 [["get_user_info"], ["get_portfolio_summary"]],
                 [call("get_user_info")], 1,
                 numbers=[num("cash", cash(), MONEY)]),
        question("lookup", "What stocks do I currently own?",
                 [["get_holdings"], ["get_portfolio_summary"]],
                 [call("get_holdings")], 1,
                 contains=all_tickers),
        question("lookup", "How many shares of AAPL do I own and what is my average cost basis?",
                 [["get_holding_by_ticker"], ["get_holdings"], ["get_portfolio_summary"]],
                 [call("get_holding_by_ticker", ticker="AAPL")], 1,
                 numbers=[num("shares", shares_owned("AAPL"), EXACT),
                          num("cost_basis", get_holding_by_ticker("AAPL")["avg_cost_basis"], MONEY)]),
        question("lookup", "What is the current price of NVDA?",
                 [["get_stock_price"]],
                 [call("get_stock_price", ticker="NVDA")], 1,
                 numbers=[num("price", price("NVDA"), MONEY)]),
        question("lookup", "What is the 52-week high and low for GOOGL?",
                 [["get_stock_price"]],
                 [call("get_stock_price", ticker="GOOGL")], 1,
                 numbers=[num("year_high", googl["year_high"], MONEY),
                          num("year_low", googl["year_low"], MONEY)]),
        question("lookup", "What is my portfolio's sector allocation in percent?",
                 [["get_sector_allocation"]],
                 [call("get_sector_allocation")], 1,
                 numbers=[num(s["sector"], s["percentage"], PCT) for s in sectors]),
        question("lookup", "What is my total portfolio value?",
                 [["get_portfolio_summary"]],
                 [call("get_portfolio_summary")], 1,
                 numbers=[num("portfolio_value", summary["portfolio_value"], MONEY)]),
        question("lookup", "Show me the option chain for TSLA.",
                 [["get_option_chain"]],
                 [call("get_option_chain", ticker="TSLA")], 1,
                 numbers=[num("nearest_call_strike", tsla_chain["calls_near_atm"][0]["strike"], EXACT)],
                 contains=["TSLA"]),
        question("lookup", "What is MSFT's percentage price change over the last 3 months?",
                 [["get_historical_data"]],
                 [call("get_historical_data", ticker="MSFT", period="3mo")], 1,
                 numbers=[num("pct_change", pct_change("MSFT", "3mo"), PCT)]),
        question("lookup", "What was my most recent trade, and at what price per share?",
                 [["get_recent_transactions"]],
                 [call("get_recent_transactions")], 1,
                 numbers=[num("price_per_share", last_trade["price_per_share"], MONEY)],
                 contains=[last_trade["ticker"]]),
    ]


# ── Educational (should be answered with NO tools) ──────────────────────────

def build_educational():
    items = [
        ("What is the difference between a call option and a put option?", ["call", "put"]),
        ("What is a covered call strategy and when would you use it?", ["covered", "call"]),
        ("Explain what P/E ratio means in simple terms.", ["earnings"]),
        ("What is dollar-cost averaging?", ["average"]),
        ("What is the difference between a market order and a limit order?", ["market", "limit"]),
        ("What does implied volatility tell you about an option?", ["volatility"]),
    ]
    questions = []
    for text, keywords in items:
        q = question("educational", text, [], [], 0, contains=keywords)
        q["expect_no_tools"] = True
        questions.append(q)
    return questions


# ── Multi-step templates (2+ tool calls each) ───────────────────────────────

CASH_AND_PRICE = [["get_user_info", "get_stock_price"], ["get_portfolio_summary", "get_stock_price"]]


def build_afford():
    questions = []
    for ticker in ["META", "NFLX", "AMD", "KO", "DIS", "COST", "V", "WMT"]:
        questions.append(question(
            "afford",
            f"How many whole shares of {ticker} could I buy with my available cash at its current price?",
            CASH_AND_PRICE,
            [call("get_user_info"), call("get_stock_price", ticker=ticker)], 2,
            numbers=[num("whole_shares", math.floor(cash() / price(ticker)), EXACT)],
        ))
    return questions


def build_cash_after_buy():
    cases = [(10, "META"), (50, "NFLX"), (5, "AMD"), (100, "KO"),
             (40, "DIS"), (3, "COST"), (20, "V"), (60, "WMT")]
    questions = []
    for n, ticker in cases:
        questions.append(question(
            "cash_after_buy",
            f"If I bought {n} shares of {ticker} at the current price, how much cash would I have left? "
            "This is hypothetical, do not place a trade.",
            CASH_AND_PRICE,
            [call("get_user_info"), call("get_stock_price", ticker=ticker)], 2,
            numbers=[num("cash_left", cash() - n * price(ticker), MONEY)],
        ))
    return questions


def build_sell_to_buy():
    pairs = [("TSLA", "KO"), ("AAPL", "DIS"), ("MSFT", "WMT"), ("GOOGL", "NFLX"),
             ("JPM", "V"), ("JNJ", "META"), ("AMZN", "AMD"), ("NVDA", "COST")]
    questions = []
    for held, target in pairs:
        proceeds = shares_owned(held) * price(held)
        questions.append(question(
            "sell_to_buy",
            f"If I sold all my {held} shares at the current price, what would the sale proceeds be, and how "
            f"many whole shares of {target} could I buy with just those proceeds? This is hypothetical, "
            "do not place a trade.",
            [["get_holding_by_ticker", "get_stock_price"],
             ["get_holdings", "get_stock_price"],
             ["get_portfolio_summary", "get_stock_price"]],
            [call("get_holding_by_ticker", ticker=held),
             call("get_stock_price", ticker=held),
             call("get_stock_price", ticker=target)], 2,
            numbers=[num("proceeds", proceeds, MONEY),
                     num("whole_shares", math.floor(proceeds / price(target)), EXACT)],
        ))
    return questions


def build_history_compare():
    cases = [("AAPL", "MSFT", "3mo"), ("NVDA", "AMD", "6mo"), ("GOOGL", "META", "1mo"),
             ("JPM", "V", "1y"), ("KO", "WMT", "3mo"), ("TSLA", "AMZN", "6mo"),
             ("DIS", "NFLX", "1y"), ("JNJ", "COST", "1mo")]
    questions = []
    for a, b, period in cases:
        questions.append(question(
            "history_compare",
            f"Which performed better over the last {PERIOD_WORDS[period]}, {a} or {b}? "
            "Give the percentage price change of each.",
            [["get_historical_data"]],
            [call("get_historical_data", ticker=a, period=period),
             call("get_historical_data", ticker=b, period=period)], 2,
            numbers=[num(f"{a}_pct_change", pct_change(a, period), PCT),
                     num(f"{b}_pct_change", pct_change(b, period), PCT)],
        ))
    return questions


def build_pe_and_shares():
    pairs = [("AAPL", "MSFT"), ("GOOGL", "AMZN"), ("NVDA", "TSLA"),
             ("JPM", "JNJ"), ("MSFT", "NVDA"), ("AAPL", "GOOGL")]
    questions = []
    for a, b in pairs:
        comparison = compare_stocks(a, b)
        questions.append(question(
            "pe_and_shares",
            f"What are the trailing P/E ratios of {a} and {b}, and how many shares of each do I own?",
            [["compare_stocks", "get_holdings"],
             ["compare_stocks", "get_holding_by_ticker"],
             ["compare_stocks", "get_portfolio_summary"]],
            [call("compare_stocks", ticker_1=a, ticker_2=b), call("get_holdings")], 2,
            numbers=[num(f"{a}_pe", comparison["stock_1"]["trailing_pe"], MONEY),
                     num(f"{b}_pe", comparison["stock_2"]["trailing_pe"], MONEY),
                     num(f"{a}_shares", shares_owned(a), EXACT),
                     num(f"{b}_shares", shares_owned(b), EXACT)],
        ))
    return questions


def build_last_purchase():
    transactions = get_recent_transactions()   # newest first
    questions = []
    for ticker in ["AAPL", "MSFT", "GOOGL", "NVDA", "BND", "TSLA"]:
        last_buy = next(t for t in transactions if t["ticker"] == ticker and t["action"] == "BUY")
        questions.append(question(
            "last_purchase",
            f"What price per share did I pay on my most recent {ticker} purchase, and what is {ticker} "
            "trading at now?",
            [["get_recent_transactions", "get_stock_price"],
             ["get_recent_transactions", "get_portfolio_summary"]],
            [call("get_recent_transactions"), call("get_stock_price", ticker=ticker)], 2,
            numbers=[num("purchase_price", last_buy["price_per_share"], MONEY),
                     num("current_price", price(ticker), MONEY)],
        ))
    return questions


def build_price_and_news():
    questions = []
    for ticker in ["META", "AMD", "DIS", "COST"]:
        questions.append(question(
            "price_and_news",
            f"What is {ticker} trading at right now, and what are the latest headlines about it?",
            [["get_stock_price", "get_stock_news"]],
            [call("get_stock_price", ticker=ticker), call("get_stock_news", ticker=ticker)], 2,
            numbers=[num("price", price(ticker), MONEY)],
        ))
    return questions


def build_closest_to_high():
    groups = [("AAPL", "MSFT", "NVDA"), ("GOOGL", "META", "AMZN"), ("JPM", "V", "KO"),
              ("TSLA", "AMD", "NFLX"), ("JNJ", "COST", "WMT"), ("DIS", "KO", "NFLX")]
    questions = []
    for group in groups:
        gaps = {}
        for ticker in group:
            data = get_stock_price(ticker)
            gaps[ticker] = (data["year_high"] - data["current_price"]) / data["year_high"] * 100
        winner = min(gaps, key=gaps.get)
        a, b, c = group
        questions.append(question(
            "closest_to_high",
            f"Among {a}, {b} and {c}, which is trading closest to its 52-week high, and how far below "
            "that high is it, as a percentage of the high?",
            [["get_stock_price"], ["compare_stocks"]],
            [call("get_stock_price", ticker=t) for t in group], 2,
            numbers=[num("gap_pct", gaps[winner], PCT)],
            contains=[winner],
        ))
    return questions


def build_sector_and_cash():
    allocation = {s["sector"]: s["percentage"] for s in get_sector_allocation()["sector_allocation"]}
    questions = []
    for sector in ["Technology", "Financials", "Healthcare", "ETF"]:
        questions.append(question(
            "sector_and_cash",
            f"What percentage of my invested portfolio is in the {sector} sector, and what is my cash balance?",
            [["get_sector_allocation", "get_user_info"], ["get_sector_allocation", "get_portfolio_summary"]],
            [call("get_sector_allocation"), call("get_user_info")], 2,
            numbers=[num("sector_pct", allocation[sector], PCT), num("cash", cash(), MONEY)],
        ))
    return questions


def build_shares_and_strike():
    questions = []
    for ticker in ["AAPL", "MSFT", "NVDA", "TSLA"]:
        chain = get_option_chain(ticker)
        questions.append(question(
            "shares_and_strike",
            f"How many shares of {ticker} do I own, and which call strike is closest to the current price "
            "for the nearest expiration?",
            [["get_holding_by_ticker", "get_option_chain"],
             ["get_holdings", "get_option_chain"],
             ["get_portfolio_summary", "get_option_chain"]],
            [call("get_holding_by_ticker", ticker=ticker), call("get_option_chain", ticker=ticker)], 2,
            numbers=[num("shares", shares_owned(ticker), EXACT),
                     num("strike", chain["calls_near_atm"][0]["strike"], EXACT)],
        ))
    return questions


def build_cash_share():
    summary = get_portfolio_summary()
    cash_pct = summary["cash_balance"] / summary["portfolio_value"] * 100
    questions = []
    for ticker in ["NFLX", "AMD", "COST", "META"]:
        questions.append(question(
            "cash_share",
            f"What percentage of my total portfolio value is cash, and how many whole shares of {ticker} "
            "would that cash buy at the current price?",
            [["get_portfolio_summary", "get_stock_price"]],
            [call("get_portfolio_summary"), call("get_stock_price", ticker=ticker)], 2,
            numbers=[num("cash_pct", cash_pct, PCT),
                     num("whole_shares", math.floor(cash() / price(ticker)), EXACT)],
        ))
    return questions


def build_change_and_value():
    cases = [("AAPL", "1mo"), ("NVDA", "3mo"), ("JPM", "6mo"), ("VTI", "1y"), ("TSLA", "1mo"), ("BND", "1y")]
    questions = []
    for ticker, period in cases:
        questions.append(question(
            "change_and_value",
            f"How much has {ticker} changed in percent over the last {PERIOD_WORDS[period]}, and what is my "
            f"{ticker} position worth at the current price?",
            [["get_historical_data", "get_holding_by_ticker", "get_stock_price"],
             ["get_historical_data", "get_holdings", "get_stock_price"],
             ["get_historical_data", "get_portfolio_summary"]],
            [call("get_historical_data", ticker=ticker, period=period),
             call("get_holding_by_ticker", ticker=ticker),
             call("get_stock_price", ticker=ticker)], 2,
            numbers=[num("pct_change", pct_change(ticker, period), PCT),
                     num("position_value", shares_owned(ticker) * price(ticker), MONEY)],
        ))
    return questions


# ── Trades (two turns: request, then confirmation) ──────────────────────────

def build_trades():
    # (request, trade arguments, keywords expected when the trade must be refused)
    cases = [
        ("Buy 5 shares of MSFT.", dict(action="BUY", ticker="MSFT", shares=5), None),
        ("Buy 10 shares of AAPL.", dict(action="BUY", ticker="AAPL", shares=10), None),
        ("Buy 20 shares of KO.", dict(action="BUY", ticker="KO", shares=20), None),
        ("Buy 3 shares of JPM.", dict(action="BUY", ticker="JPM", shares=3), None),
        ("Sell 3 shares of TSLA.", dict(action="SELL", ticker="TSLA", shares=3), None),
        ("Sell 10 shares of BND.", dict(action="SELL", ticker="BND", shares=10), None),
        ("Sell all 25 of my AAPL shares.", dict(action="SELL", ticker="AAPL", shares=25), None),
        ("Place a limit order to buy 5 shares of AAPL at $320.",
         dict(action="BUY", ticker="AAPL", shares=5, order_type="LIMIT", limit_price=320), None),
        ("Buy 1000 shares of NVDA.", dict(action="BUY", ticker="NVDA", shares=1000),
         ["500", "maximum", "too large"]),
        ("Buy 10 shares of GME.", dict(action="BUY", ticker="GME", shares=10),
         ["restricted"]),
        ("Sell 50 shares of TSLA.", dict(action="SELL", ticker="TSLA", shares=50),
         ["only", "8 shares", "cannot", "can't"]),
        ("Buy 100 shares of MSFT.", dict(action="BUY", ticker="MSFT", shares=100),
         ["insufficient", "not enough", "exceed"]),
    ]

    questions = []
    for request, trade, refusal_keywords in cases:
        ticker = trade["ticker"]

        # Ground truth: run the trade for real on a fresh database and read the result.
        reset_database()
        result = execute_trade(**trade)
        holding = get_holding_by_ticker(ticker)
        state = {
            "cash": cash(),
            "holdings": {ticker: 0 if "error" in holding else holding["shares"]},
        }
        reset_database()

        should_execute = refusal_keywords is None
        assert ("error" not in result) == should_execute, f"Unexpected trade result for: {request}"

        verb = trade["action"].lower()
        questions.append({
            "category": "trade",
            "min_tool_calls": 1,
            "turns": [request, CONFIRM_MESSAGE],
            "expected_tools": [["execute_trade"]] if should_execute else [],
            "forbidden_tools": [],
            "expect_no_tools": False,
            "numbers": [],
            "contains": refusal_keywords or [],
            "contains_match": "any",
            "state": state,
            "oracle": [
                {"calls": [call("get_stock_price", ticker=ticker)],
                 "reply": f"I will {verb} {trade['shares']} shares of {ticker}. Do you want me to proceed?"},
                {"calls": [call("execute_trade", **trade)], "reply": None},
            ],
        })
    return questions


def build_all():
    reset_database()

    questions = (
        build_lookups() + build_educational()
        + build_afford() + build_cash_after_buy() + build_sell_to_buy()
        + build_history_compare() + build_pe_and_shares() + build_last_purchase()
        + build_price_and_news() + build_closest_to_high() + build_sector_and_cash()
        + build_shares_and_strike() + build_cash_share() + build_change_and_value()
        + build_trades()
    )

    # Give each question an id, and alternate dev/test inside every category so
    # both halves cover the same kinds of question.
    seen = Counter()
    for q in questions:
        seen[q["category"]] += 1
        index = seen[q["category"]]
        q["id"] = f"{q['category']}-{index:02d}"
        q["split"] = "dev" if index % 2 == 1 else "test"

    return questions


if __name__ == "__main__":
    questions = build_all()
    assert len(questions) == 100, f"Expected 100 questions, built {len(questions)}"

    offline.write_jsonl(offline.QUESTIONS_PATH, questions)

    print(f"Wrote {len(questions)} questions to {offline.QUESTIONS_PATH}")
    for category, count in Counter(q["category"] for q in questions).items():
        print(f"  {category:<18} {count}")
    multi_step = sum(1 for q in questions if q["min_tool_calls"] >= 2 or len(q["turns"]) > 1)
    print(f"Multi-step (2+ tool calls, or a multi-turn trade): {multi_step}")
